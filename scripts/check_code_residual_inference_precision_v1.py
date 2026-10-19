"""Compare cached/full inference and readout precision on fixed original source controls."""
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F

from slm_lab.code_residual_inference_v1 import load_residual_policy
from slm_lab.code_source_generation_diagnostic_v1 import select_controls, reference_agreement
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest

ROOT = Path(__file__).resolve().parents[1]


def describe(logits, gold, tokenizer):
    values = logits.float().cpu()
    maximum = values.max()
    logps = values.log_softmax(-1)
    top = values.topk(8).indices.tolist()
    return dict(argmax_token_id=int(values.argmax()), gold_token_id=gold,
        gold_strict_greater_rank=int((values > values[gold]).sum()) + 1,
        gold_tied_values=int((values == values[gold]).sum()), maximum_tied_values=int((values == maximum).sum()),
        gold_probability=float(logps[gold].exp()),
        top=[dict(id=i, token=tokenizer.decode([i], skip_special_tokens=False), logit=float(values[i])) for i in top])


@torch.no_grad()
def main():
    reports = ROOT / 'reports/code-residual-inference-precision-v1'
    if reports.exists():
        raise ValueError('Preserve previous inference diagnostic evidence')
    # No training writer is live; these diagnostics allocate no new model state.
    # Original future-stage25GiB preflight is untouched and the resumption
    # controller also waits for idle GPUs before starting its writer.
    if Path('/proc/2510561').exists() or shutil.disk_usage(ROOT).free < 16 * 1024**3 + 32 * 1024**2:
        raise ValueError('Original writer must be terminal; bounded32MiB report above16GiB reserve required')
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0] < 20 * 1024**3:
        raise ValueError('Initially20GiB free on a BF16-capable diagnostic GPU required')
    torch.cuda.set_per_process_memory_fraction(.35, device)
    pilot = ROOT / 'reports/code-residual-lr-pilot-v1'
    trained = json.loads((pilot / 'results.json').read_text())
    plan = json.loads((pilot / 'plan.json').read_text())
    selection_path = pilot / 'selected-original-rows.json'
    selected = select_controls(json.loads(selection_path.read_text()), budget=1024)
    checkpoint, base = Path(trained['best_complete_checkpoint']), Path(plan['source_checkpoint'])
    token_path = checkpoint.parent / 'tokenizer.json'
    packs = {role: CompleteCodePack(ROOT / 'data/codenet-durable-pack-v2', role,
        purpose='training' if role == 'train' else 'audit') for role in ['train', 'validation']}
    paths = [Path(__file__), pilot / 'plan.json', pilot / 'results.json', selection_path, checkpoint, base,
        token_path, ROOT / 'src/slm_lab/code_residual_inference_v1.py',
        ROOT / 'src/slm_lab/code_residual_branch_v1.py', ROOT / 'src/slm_lab/code_source_generation_diagnostic_v1.py',
        ROOT / 'src/slm_lab/model.py', ROOT / 'src/slm_lab/training.py',
        ROOT / 'src/slm_lab/codenet_checkpoint_v1.py', ROOT / 'src/slm_lab/codenet_token_pack_v1.py',
        ROOT / 'reports/code-residual-source-generation-v1/plan.json',
        ROOT / 'reports/code-residual-source-generation-v1/results.json',
        packs['train'].folder / 'manifest.json']
    bound = {str(p): file_hash(p) for p in paths}
    bound.update({str(packs['train'].folder / name): h for name, h in packs['train'].manifest['files_sha256'].items()})
    guard = FileBindings(bound)
    reports.mkdir()

    def status(message, **fields):
        atomic_json(reports / 'status.json', dict(status=message,
            updated_at=datetime.now(timezone.utc).isoformat(), optimizer_updates=0,
            candidate_promoted=False, goal_achieved=False, **fields))

    atomic_json(reports / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound, original_source_controls=selected,
        modes=['BF16_default', 'BF16_backbone_FP32_readout', 'FP32_all'],
        generation_budget=64, all_fresh_caches_per_call=True,
        comparisons=['full forward', 'cached prefill twice', 'cached generate1', 'uncached generate1', 'cached generate64'],
        minimum_free_home_bytes=16 * 1024**3, maximum_report_bytes=32 * 1024**2,
        device=str(device), maximum_GPU_memory_fraction=.35,
        original_training_writer_terminal=True, next_original_stage_storage_checks_unchanged=True,
        original_primary_benchmark_outputs_and_scoring_unchanged=True,
        global_model_or_generation_source_changes=False, optimizer_updates=0, goal_achieved=False,
        scope='Eight already frozen short train/internal-validation source controls, no benchmark cases or model selection.64-token runs diagnose early dynamics only; they are not complete coding tests. A process-local removable hook changes only readout arithmetic for one diagnostic mode. Parameter state must remain exact. Strict-greater rank can be1 when several tokens tie; it is not the greedy-token identity.'))
    try:
        status('loading exact frozen own857M code route for inference-only precision comparisons')
        model, metadata = load_residual_policy(checkpoint, base, token_path,
            expected_checkpoint_sha256=trained['best_complete_checkpoint_sha256'],
            expected_base_sha256=plan['source_checkpoint_sha256'],
            expected_plan_sha256=trained['run_plan_sha256'], device=str(device))
        tokenizer = Tokenizer.from_file(str(token_path))
        before = state_digest(model.state_dict())
        records, vectors = [], []

        def fp32_readout(module, inputs, output):
            with torch.autocast('cuda', enabled=False):
                return F.linear(inputs[0].float(), module.weight.float())

        for mode in ['BF16_default', 'BF16_backbone_FP32_readout', 'FP32_all']:
            handle = model.lm_head.register_forward_hook(fp32_readout) if mode == 'BF16_backbone_FP32_readout' else None
            amp = (lambda: nullcontext()) if mode == 'FP32_all' else (lambda: torch.autocast('cuda', dtype=torch.bfloat16))
            try:
                for row in selected:
                    guard.check()
                    ids, labels = packs[row['role']].sequence(row['local_position'])
                    if hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() != row['full_input_ids_sha256']:
                        raise ValueError('Exact complete original source token identity required')
                    start = next(i for i, value in enumerate(labels) if value != -100)
                    reference = ids[start:]
                    prompt = torch.tensor([ids[:start]], device=device)
                    with amp():
                        out = model(prompt)
                        full = out['logits'][0, -1].float().cpu()
                        del out
                        out = model.prefill(prompt)
                        cached = out['logits'][0, -1].float().cpu()
                        del out
                        out = model.prefill(prompt)
                        repeated = out['logits'][0, -1].float().cpu()
                        del out
                        generated1 = int(model.generate(prompt, max_new_tokens=1, use_cache=True)[0, -1])
                        uncached1 = int(model.generate(prompt, max_new_tokens=1, use_cache=False)[0, -1])
                        generated = model.generate(prompt, max_new_tokens=64, use_cache=True,
                            eos_token_id=tokenizer.token_to_id('<eos>'))[0, start:].tolist()
                    record = dict(mode=mode, source_id=row['id'], role=row['role'], language=row['language'],
                        prompt_tokens=start, full=describe(full, reference[0], tokenizer),
                        cached=describe(cached, reference[0], tokenizer), repeated=describe(repeated, reference[0], tokenizer),
                        cached_full_max_abs_error=float((full - cached).abs().max()),
                        repeated_cached_max_abs_error=float((cached - repeated).abs().max()),
                        generated1_token=generated1, uncached1_token=uncached1,
                        cached_argmax_matches_generated1=int(cached.argmax()) == generated1,
                        full_argmax_matches_uncached1=int(full.argmax()) == uncached1,
                        generation_budget=64, generated_token_ids=generated,
                        raw_text=tokenizer.decode(generated[:-1] if generated and generated[-1] == 2 else generated,
                            skip_special_tokens=False), **reference_agreement(generated, reference, 2))
                    records.append(record)
                    vectors.append(dict(mode=mode, source_id=row['id'], full=full, cached=cached, repeated=repeated))
                    status('checking every frozen original source in all precision modes', completed=len(records), planned=24,
                        mode=mode, source_id=row['id'])
                    print('precision_source', len(records), mode, row['id'], flush=True)
            finally:
                if handle is not None:
                    handle.remove()
        guard.check(full=True)
        after = state_digest(model.state_dict())
        if before != after or model.lm_head._forward_hooks:
            raise ValueError('No hook or changed parameter may survive the diagnostic')
        torch.save(vectors, reports / 'full-next-token-logits.pt')
        atomic_json(reports / 'records.json', dict(records=records))
        if sum(p.stat().st_size for p in reports.iterdir() if p.is_file()) > 32 * 1024**2:
            raise ValueError('Declared32MiB report budget exceeded')
        summaries = []
        for mode in ['BF16_default', 'BF16_backbone_FP32_readout', 'FP32_all']:
            rows = [r for r in records if r['mode'] == mode]
            summaries.append(dict(mode=mode, controls=len(rows),
                maximum_cached_full_logit_error=max(r['cached_full_max_abs_error'] for r in rows),
                maximum_repeated_prefill_logit_error=max(r['repeated_cached_max_abs_error'] for r in rows),
                cached_argmax_generate1_agreements=sum(r['cached_argmax_matches_generated1'] for r in rows),
                full_argmax_uncached1_agreements=sum(r['full_argmax_matches_uncached1'] for r in rows),
                maximum_logit_ties_controls=sum(r['cached']['maximum_tied_values'] > 1 for r in rows),
                first_reference_token_matches=sum(r['first_reference_token_match'] for r in rows),
                terminal_EOS_within64=sum(r['terminal_EOS'] for r in rows)))
        result = dict(completed_at=datetime.now(timezone.utc).isoformat(), summaries=summaries,
            source_files_sha256=bound, records_sha256=file_hash(reports / 'records.json'),
            full_logits_sha256=file_hash(reports / 'full-next-token-logits.pt'),
            original_state_sha256=before, final_state_sha256=after,
            TF32_allowed=torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            global_model_or_generation_sources_modified=False, optimizer_updates=0,
            primary_benchmark_scores_modified=False, candidate_promoted=False, goal_achieved=False,
            scope='Source-only early inference arithmetic/cache diagnostic.64-token termination is not whole-program correctness; all original primary capability failures and broader goals remain.')
        atomic_json(reports / 'results.json', result)
        status('complete source-only inference precision/cache diagnostic', summaries=summaries)
    except Exception as error:
        status('diagnostic stopped; preserve evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
