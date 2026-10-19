"""Assess all retained conditioned-recurrence policies on native C++ functions."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from slm_lab.conditioned_recurrent_model import ConditionedRecurrentLM
from slm_lab.code_tasks import TASKS, request
from slm_lab.corpus import file_hash
from slm_lab.cpp_evaluation import extract
from slm_lab.cpp_tasks import SIGNATURES, reference
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.native_code import execute, efficiency, toolchain
from slm_lab.recurrent_ablation import state_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preceding-pid', type=int, required=True)
    parser.add_argument('--preceding-start-ticks', required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    reports = ROOT / 'reports/conditioned-recurrent-cpp-v1'
    if reports.exists(): raise ValueError('Preserve the original native C++ architecture assessment')
    reports.mkdir()
    def status(state, **fields):
        atomic_json(reports / 'status.json', {'status': state,
            'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **fields})
    while True:
        p = Path('/proc') / str(args.preceding_pid)
        if not p.exists(): break
        stat = (p / 'stat').read_text().split(') ', 1)[1].split()
        command = (p / 'cmdline').read_bytes().split(b'\0')
        if stat[19] != args.preceding_start_ticks or stat[0] in {'Z', 'X'}: break
        if b'scripts/train_conditioned_recurrent_ablation.py' not in command:
            raise ValueError('Preceding actor is not the owned conditioned comparison')
        status('waiting for verified live conditioned architecture comparison', preceding_pid=args.preceding_pid,
               preceding_start_ticks=args.preceding_start_ticks)
        time.sleep(15)
    source = ROOT / 'reports/conditioned-recurrent-code-ablation-v1'
    if json.loads((source / 'status.json').read_text())['status'] != 'full-language recurrence comparison completed; broad capability unverified':
        raise ValueError('Conditioned comparison did not complete; do not assess a partial policy')
    comparison_path = source / 'comparison.json'; comparison_sha = file_hash(comparison_path)
    comparison = json.loads(comparison_path.read_text()); plan = comparison['plan']
    for name, digest in plan['source_runtime']['files_sha256'].items():
        if file_hash(ROOT / name) != digest: raise ValueError('Original conditioned trial runtime changed')
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']]
        + [ROOT / name for name in plan['source_runtime']['files_sha256']] + [Path(__file__)]))
    runtime = capture(files); tools = toolchain(); device = 'cuda:3'
    torch.cuda.set_per_process_memory_fraction(.2, 3)
    base = ROOT / 'checkpoints/extended-curriculum-v1/foundation-warmup/model.pt'
    if file_hash(base) != plan['base_checkpoint_sha256']: raise ValueError('Original owned warmup changed')
    tokenizer_path = base.parent / 'tokenizer.json'
    if file_hash(tokenizer_path) != plan['tokenizer_sha256']: raise ValueError('Own tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    references = {}
    for task in TASKS:
        measured = execute(request(task, 'C++', reference(task[0]), True))
        if not measured.get('functional_passed') or not measured.get('efficiency_completed'):
            raise ValueError('Original native C++ reference failed: ' + task[0])
        references[task[0]] = measured
    atomic_json(reports / 'references.json', {'toolchain': tools, 'results': references,
        'scope': 'Original local native function/reference verification only; not model capability.'})
    outcomes = {}
    for name in ['calibrated-initial-one-pass', *comparison['outcomes']]:
        status('loading isolated own conditioned policy for native C++ assessment', branch=name)
        if name == 'calibrated-initial-one-pass':
            original = torch.load(base, weights_only=True, map_location='cpu', mmap=True)
            model = ConditionedRecurrentLM(ModelConfig(**original['config']), recurrences=1)
            model.load_owned_base(original['model']); del original
            scales = plan['training_only_norm_calibration']
            with torch.no_grad():
                for i, (attention, feedforward) in enumerate(zip(model.recurrence_attn_post, model.recurrence_ffn_post)):
                    attention.weight.fill_(scales['attention_rms'][i]); feedforward.weight.fill_(scales['feedforward_rms'][i])
                model.recurrence_output_norm.weight.fill_(scales['feedforward_rms'][-1])
            expected = plan['initial_conditioned_policy_state_sha256']; checkpoint_sha = None
        else:
            branch = comparison['outcomes'][name]; checkpoint = Path(branch['checkpoint'])
            if file_hash(checkpoint) != branch['checkpoint_sha256']: raise ValueError('Own branch checkpoint changed')
            original = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
            if original['architecture'] != 'own-input-conditioned-normalized-recurrence-v1':
                raise ValueError('Unexpected own branch architecture')
            model = ConditionedRecurrentLM(ModelConfig(**original['config']), recurrences=original['recurrences'],
                core_start=original['core_start'], core_end=original['core_end'], max_recurrences=original['max_recurrences'])
            model.load_state_dict(original['model']); expected = branch['model_state_sha256']
            checkpoint_sha = branch['checkpoint_sha256']; del original
        if state_digest(model.state_dict()) != expected: raise ValueError('Actual reconstructed inference state differs')
        model.to(device).eval(); rows = []
        for task in TASKS:
            function, _, description, scaling, _, _ = task
            signature = SIGNATURES[function]
            prompt = f'User: Write an ISO C++17 function with signature {signature}. {description}\nInt means long long; Ints means std::vector<Int>. Strings use std::u32string/char32_t, one element per Unicode BMP character. Use std::nullopt for missing optional results. Standard library headers and these aliases are supplied. No main function, nonstandard headers, OS APIs or threads. Return only the complete function and helpers.\nAssistant:'
            ids = torch.tensor([[tokenizer.token_to_id('<bos>')] + tokenizer.encode(prompt).ids], device=device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated = model.generate(ids, max_new_tokens=512, eos_token_id=tokenizer.token_to_id('<eos>'))
            raw = tokenizer.decode(generated[0, ids.shape[1]:].tolist()); code = extract(raw)
            measured = execute(request(task, 'C++', code)); baseline = references[function]
            passed, measurement = efficiency(measured, baseline)
            rows.append({'task': function, 'signature': signature, 'raw_response': raw, 'code': code,
                'candidate': measured, 'efficiency_stress_passed': passed, 'efficiency_measurement': measurement})
            status('assessing all original native C++ algorithm functions', branch=name, completed_tasks=len(rows))
            print(json.dumps({'branch': name, 'task': function, 'syntax': measured['syntax_valid'],
                              'functional': measured['functional_passed']}), flush=True)
        result = {'branch': name, 'model_state_sha256': expected, 'checkpoint_sha256': checkpoint_sha,
            'recurrences': model.recurrences, 'tasks': len(rows), 'generation_budget': 512,
            'prompt_mode': 'native instruction', 'syntax_passes': sum(r['candidate']['syntax_valid'] for r in rows),
            'functional_passes': sum(r['candidate']['functional_passed'] for r in rows),
            'efficiency_passes': sum(r['efficiency_stress_passed'] for r in rows), 'results': rows,
            'toolchain': tools, 'evaluation_provenance': runtime, 'goal_achieved': False}
        atomic_json(reports / (name + '.json'), result); outcomes[name] = result
        del model; gc.collect(); torch.cuda.empty_cache()
        if capture(files) != runtime or file_hash(comparison_path) != comparison_sha:
            raise ValueError('Native assessment source/runtime/comparison changed')
    atomic_json(reports / 'comparison.json', {'outcomes': outcomes, 'source_comparison_sha256': comparison_sha,
        'evaluation_provenance': runtime, 'automatic_policy_promotion': False, 'goal_achieved': False,
        'scope': 'Twelve local C++17 algorithm contracts per policy, complete native hidden/empirical stress tests. Not OS/network APIs, original repository modules, universal C++ or proof of asymptotic optimality.'})
    status('native C++ conditioned architecture assessment completed; broad capability unverified')


if __name__ == '__main__': main()
