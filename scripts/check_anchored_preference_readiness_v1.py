"""Full own-model anchored preference backward check, with zero weight updates."""
import ast
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from slm_lab.anchored_preference_gradients_v1 import gradients, score
from slm_lab.code_quality import execute, instruction_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.instruction_generalization_trial_v3 import complete_encoding
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_ablation import language_parameters, state_digest
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / 'reports/anchored-preference-full-model-readiness-v1.json'
    if report.exists():
        raise ValueError('Preserve existing full-model readiness evidence')
    torch.set_num_threads(1)
    trained = json.loads((ROOT / 'reports/instruction-generalization-probe-v3/training.json').read_text())
    plan = json.loads((ROOT / 'reports/instruction-generalization-probe-v3/plan.json').read_text())
    checkpoint = Path(trained['candidate_checkpoint'])
    if file_hash(checkpoint) != trained['candidate_checkpoint_sha256']:
        raise ValueError('Completed own-policy checkpoint changed')
    source = Path(plan['input_manifest']); data = source.parent / 'verified_python.jsonl'
    manifest = json.loads(source.read_text())
    if file_hash(source) != plan['input_manifest_sha256'] or file_hash(data) != manifest['files_sha256'][data.name]:
        raise ValueError('Original admitted public source differs')
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
             ROOT / 'src/slm_lab/anchored_preference_gradients_v1.py',
             ROOT / 'src/slm_lab/instruction_generalization_trial_v3.py',
             ROOT / 'src/slm_lab/recurrent_ablation.py']
    runtime = capture(files)
    metadata = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    config = ModelConfig(**metadata['config']); tokenizer = checkpoint.parent / 'tokenizer.json'
    if file_hash(tokenizer) != metadata['tokenizer_sha256'] or metadata['steps'] != 1500:
        raise ValueError('Actual own tokenizer/training budget differs')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), config, data.parent, 448)
    selected = None
    for row in jsonl(data):
        if row['variant_index'] != 0 or len(encoder.tokens(row['response'])) > 255:
            continue
        tree = ast.parse(instruction_code(row['response']))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == row['function'])
        function.body = [ast.Return(value=ast.Constant(value=None))]
        bad = '```python\n' + ast.unparse(ast.fix_missing_locations(tree)) + '\n```'

        def check(text):
            return execute({'language': 'Python', 'code': instruction_code(text), 'function': row['function'],
                            'cases': row['verification_cases'], 'stress': []})

        good_result, bad_result = check(row['response']), check(bad)
        if good_result['functional_passed'] and bad_result.get('error') == 'Hidden case mismatch':
            selected = {**row, 'chosen': row['response'], 'rejected': bad}
            complete_encoding(row, encoder); complete_encoding({**row, 'response': bad}, encoder)
            break
    if selected is None:
        raise ValueError('No original public program supports this nonlearning runtime check')
    device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.3, 3)
    if torch.cuda.mem_get_info(3)[0] < 12 * 1024**3:
        raise ValueError('Insufficient currently free GPU memory for isolated backward readiness')
    model = TinyMultimodalLM(config); model.load_state_dict(metadata['model']); del metadata
    before = state_digest(model.state_dict()); count = model.parameter_counts()['total']
    if count != 706012672 or count >= 1_000_000_000:
        raise ValueError('Original own model parameter budget differs')
    parameters = language_parameters(model, False); trainable = sum(p.numel() for p in parameters)
    model.to(device); model.gradient_checkpointing = True; model.train()
    chosen, rejected = [encoder.batch([selected], field, device) for field in ['chosen', 'rejected']]
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(3)
    with torch.no_grad(), amp():
        reference_good, reference_bad = score(model, chosen), score(model, rejected)
    actual = gradients(model, chosen, rejected, reference_good, reference_bad, amp)
    if not math.isclose(actual['preference_loss'], math.log(2), abs_tol=1e-5):
        raise ValueError('Initial exact policy/reference objective differs')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0 or any(p.grad is not None for name, p in model.named_parameters()
            if name.startswith('vision.') or name == 'value_head.weight'):
        raise ValueError('Nonfinite/missing language gradients or unintended frozen-parameter gradients')
    torch.cuda.synchronize(3)
    peak = torch.cuda.max_memory_allocated(3)
    if state_digest(model.state_dict()) != before or encoder.truncated:
        raise ValueError('Readiness altered original own model tensors or clipped data')
    if capture(files) != runtime or file_hash(data) != manifest['files_sha256'][data.name]:
        raise ValueError('Readiness implementation/runtime/source changed')
    result = {'own_checkpoint_sha256': trained['candidate_checkpoint_sha256'],
        'own_total_parameters': count, 'language_trainable_parameters': trainable,
        'original_model_state_sha256': before, 'all_model_tensors_unchanged': True,
        'optimizer_updates': 0, 'device': device, 'precision': 'BF16 forward, FP32 response log probabilities',
        'gradient_checkpointing': True, 'peak_cuda_allocated_bytes': peak,
        'language_gradient_norm_before_clipping': norm, 'objective': actual,
        'public_source_id': selected['id'], 'source_data_sha256': file_hash(data),
        'positive_execution': good_result, 'negative_execution': bad_result,
        'negative_origin': 'Deterministic return-None mutation solely for nonlearning readiness; not an own-policy generation or preference training dataset.',
        'source_runtime': runtime, 'goal_achieved': False,
        'scope': 'Full own706M forward/backward runtime verification with frozen vision/critic and exact unchanged weights. No optimizer-state allocation, learning or capability evidence.'}
    atomic_json(report, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ['source_runtime', 'positive_execution', 'negative_execution']}, indent=2))


if __name__ == '__main__':
    main()
