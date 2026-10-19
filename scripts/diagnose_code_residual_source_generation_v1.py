"""Check fixed original source controls, with no training, execution or benchmark edits."""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.code_residual_inference_v1 import load_residual_policy
from slm_lab.code_source_generation_diagnostic_v1 import OriginalBaseView, select_controls, reference_agreement
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest

ROOT = Path(__file__).resolve().parents[1]


def verified_occupied_snapshot():
    """Credit actual committed bytes from the exact recovered original writer."""
    receipt = ROOT / 'reports/curriculum-space-recovery-v3/recovery.json'
    recovered = json.loads(receipt.read_text())
    main = recovered['main']
    process = Path('/proc') / str(main['pid'])
    fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
    args = [a.decode() for a in (process / 'cmdline').read_bytes().split(b'\0') if a]
    if fields[19] != main['start_ticks'] or fields[0] in {'Z', 'X'} or args != main['argv']:
        raise ValueError('Exact recovered live original controller required for occupied-byte credit')
    for name, expected in recovered['source_files_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Recovered original frozen controller or plan changed')
    status = json.loads((ROOT / 'reports/extended-curriculum-v1/status.json').read_text())
    if status.get('stage') != 'verified-standalone-code':
        raise ValueError('Inspect capacity again after a different stage starts')
    directory = ROOT / 'checkpoints/extended-curriculum-v1/verified-standalone-code'
    candidates = sorted(directory.glob('step-*.pt'))
    if not candidates:
        return dict(path=None, bytes=0, credited_bytes=0, policy='no committed snapshot to credit')
    snapshot = candidates[-1]
    before = snapshot.stat()
    state = torch.load(snapshot, map_location='cpu', weights_only=True, mmap=True)
    if (state.get('stage') != 'sft' or state.get('world_size') != 4
            or state.get('planned_steps') != 1324 or not 0 < state.get('steps', 0) < 1324
            or state.get('data_sha256') != '9fe0a3be9c1b0057e3df02a89cc7f264d43f142e0ae003522aec69f85cda8ce8'
            or not state.get('optimizer') or 'torch_rng_state' not in state or 'cuda_rng_state' not in state):
        raise ValueError('Exact active-stage full optimizer/RNG snapshot required for capacity credit')
    digest = file_hash(snapshot)
    after = snapshot.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Committed capacity snapshot changed during verification')
    return dict(path=str(snapshot), bytes=before.st_size, sha256=digest,
        credited_bytes=min(before.st_size, 9 * 1024**3), completed_steps=state['steps'], planned_steps=1324,
        original_controller_pid=main['pid'], original_controller_start_ticks=main['start_ticks'],
        policy='only verified committed current-stage bytes credited against unchanged26GiB budget')


@torch.no_grad()
def main():
    reports = ROOT / 'reports/code-residual-source-generation-v1'
    if reports.exists():
        raise ValueError('Preserve any previous partial or complete diagnostic')
    occupied = verified_occupied_snapshot()
    reserve = 26 * 1024**3 - occupied['credited_bytes']
    if shutil.disk_usage(ROOT).free < reserve + 32 * 1024**2:
        raise ValueError('Full original26GiB writer reserve and32MiB report budget required')
    torch.set_num_threads(1)
    device = torch.device('cuda:3')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0] < 12 * 1024**3:
        raise ValueError('Initially12GiB free on a BF16-capable shared GPU required')
    torch.cuda.set_per_process_memory_fraction(.14, device)
    pilot = ROOT / 'reports/code-residual-lr-pilot-v1'
    trained = json.loads((pilot / 'results.json').read_text())
    plan = json.loads((pilot / 'plan.json').read_text())
    selection_path = pilot / 'selected-original-rows.json'
    if (file_hash(pilot / 'plan.json') != trained['run_plan_sha256']
            or file_hash(selection_path) != plan['selection_sha256']):
        raise ValueError('Exact frozen train/validation-only pilot selection required')
    selected = select_controls(json.loads(selection_path.read_text()), budget=1024)
    pack_root = ROOT / 'data/codenet-durable-pack-v2'
    packs = dict(train=CompleteCodePack(pack_root, 'train', purpose='training'),
        validation=CompleteCodePack(pack_root, 'validation', purpose='audit'))
    positions = {r['global_position']: r for r in selected}
    originals = {}
    with gzip.open(pack_root / 'original-rows.jsonl.gz', 'rt') as stream:
        for position, line in enumerate(stream):
            if position in positions:
                originals[position] = json.loads(line)
    if len(originals) != 8:
        raise ValueError('Every selected complete original source row is required')
    checkpoint = Path(trained['best_complete_checkpoint'])
    base = Path(plan['source_checkpoint'])
    token_path = checkpoint.parent / 'tokenizer.json'
    paths = [Path(__file__), ROOT / 'src/slm_lab/code_source_generation_diagnostic_v1.py',
        ROOT / 'src/slm_lab/code_residual_inference_v1.py',
        ROOT / 'src/slm_lab/code_residual_branch_v1.py', ROOT / 'src/slm_lab/code_residual_checkpoint_v1.py',
        ROOT / 'src/slm_lab/codenet_checkpoint_v1.py', ROOT / 'src/slm_lab/codenet_complete_context_v1.py',
        ROOT / 'src/slm_lab/codenet_token_pack_v1.py', ROOT / 'src/slm_lab/model.py',
        ROOT / 'src/slm_lab/training.py', ROOT / 'src/slm_lab/recurrent_ablation.py',
        ROOT / 'tests/test_code_source_generation_diagnostic_v1.py',
        ROOT / 'reports/code-source-generation-diagnostic-v1-focused-tests.log',
        pilot / 'plan.json', pilot / 'results.json', selection_path, checkpoint, base, token_path,
        pack_root / 'manifest.json', pack_root / 'admission.json']
    bound = {str(p): file_hash(p) for p in paths}
    bound.update({str(pack_root / name): h for name, h in packs['train'].manifest['files_sha256'].items()})
    guard = FileBindings(bound)
    reports.mkdir()

    def status(message, **fields):
        atomic_json(reports / 'status.json', dict(status=message,
            updated_at=datetime.now(timezone.utc).isoformat(), optimizer_updates=0,
            model_promoted=False, goal_achieved=False, **fields))

    atomic_json(reports / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound, selected_original_rows=selected,
        selection_rule='Shortest original answer per train/validation role and C++/Python/D/Java label, ties by source id; complete prompt plus1024 tokens must fit4096.',
        routes=['original_general_base', 'frozen_selected_code_residual'],
        source_controls=8, raw_generations=16, generation_budget=1024, temperature=0.,
        maximum_report_bytes=32 * 1024**2, original_writer_budget_bytes=26 * 1024**3,
        verified_occupied_original_writer_snapshot=occupied, remaining_original_writer_reserve_bytes=reserve,
        maximum_GPU_memory_fraction=.14, device=str(device), optimizer_updates=0,
        training_or_model_selection_performed=False, benchmark_questions_or_cases_loaded=False,
        scope='Source-only train/internal-validation diagnostic, deliberately short answers. Exact reference agreement is neither required for alternative valid algorithms nor a functional correctness score. Historical source/runtime and prior-exposure qualifications remain. No held-case or candidate-promotion feedback.'))
    try:
        status('loading exact frozen original base and selected residual policy')
        code, metadata = load_residual_policy(checkpoint, base, token_path,
            expected_checkpoint_sha256=trained['best_complete_checkpoint_sha256'],
            expected_base_sha256=plan['source_checkpoint_sha256'],
            expected_plan_sha256=trained['run_plan_sha256'], device=str(device))
        original = OriginalBaseView(code).eval()
        if (state_digest(original.state_dict()) != trained['original_base_state_sha256']
                or sum(p.numel() for p in original.parameters()) != 706012672
                or sum(p.numel() for p in code.parameters()) != 857007616):
            raise ValueError('Exact shared706M original base and857M residual route required')
        tokenizer = Tokenizer.from_file(str(token_path))
        encoder = CompleteCodeEncoder(tokenizer, code.config, pack_root)
        source_sequences = {}
        for selected_row in selected:
            row = originals[selected_row['global_position']]
            ids, labels = packs[selected_row['role']].sequence(selected_row['local_position'])
            reencoded = encoder.sequence(row, 'response')
            digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()
            if (row['id'] != selected_row['id'] or row['split'] != selected_row['role']
                    or row['language'] != selected_row['language'] or reencoded != (ids, labels)
                    or digest != selected_row['full_input_ids_sha256']):
                raise ValueError('Every complete original source prompt/program/EOS/mask must reencode')
            source_sequences[row['id']] = (ids, labels)
        before_code = state_digest(code.state_dict())
        before_base = state_digest(original.state_dict())
        records = []
        output = reports / 'generations.jsonl'
        with output.open('x') as stream:
            for route, model in [('original_general_base', original), ('frozen_selected_code_residual', code)]:
                for row in selected:
                    guard.check()
                    ids, labels = source_sequences[row['id']]
                    start = next(i for i, label in enumerate(labels) if label != -100)
                    reference = ids[start:]
                    prompt = torch.tensor([ids[:start]], dtype=torch.long, device=device)
                    started = time.monotonic()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        whole = model(torch.tensor([ids], device=device),
                            labels=torch.tensor([labels], device=device))
                        teacher_nll = float(whole['loss'])
                        del whole
                        initial = model.prefill(prompt)
                        first_log_probs = initial['logits'][0, -1].float().log_softmax(-1)
                        first_gold_probability = float(first_log_probs[reference[0]].exp())
                        first_gold_rank = int((first_log_probs > first_log_probs[reference[0]]).sum()) + 1
                        if not math.isfinite(teacher_nll) or not math.isfinite(first_gold_probability):
                            raise ValueError('Finite original-reference diagnostics required')
                        del first_log_probs, initial
                        generated = model.generate(prompt, max_new_tokens=1024,
                            eos_token_id=tokenizer.token_to_id('<eos>'), use_cache=True)
                    generated_ids = generated[0, prompt.shape[1]:].tolist()
                    raw = tokenizer.decode(generated_ids[:-1] if generated_ids and generated_ids[-1] == 2 else generated_ids,
                        skip_special_tokens=False)
                    record = dict(route=route, source_id=row['id'], role=row['role'],
                        language=row['language'], group_id=row['group_id'], original_input_ids_sha256=row['full_input_ids_sha256'],
                        prompt_tokens=start, reference_tokens_including_EOS=len(reference),
                        reference_text=originals[row['global_position']]['response'],
                        reference_token_ids=reference, generated_token_ids=generated_ids,
                        raw_program_text=raw, raw_program_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        generated_tokens=len(generated_ids), generation_budget=1024,
                        generation_seconds=time.monotonic()-started,
                        complete_reference_teacher_NLL=teacher_nll,
                        first_reference_token_prefill_probability=first_gold_probability,
                        first_reference_token_prefill_rank=first_gold_rank,
                        **reference_agreement(generated_ids, reference, 2))
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stream.flush()
                    records.append(record)
                    if output.stat().st_size > 31 * 1024**2 or shutil.disk_usage(ROOT).free < 512 * 1024**2:
                        raise ValueError('Bounded diagnostic output/storage exceeded')
                    status('generating all fixed original source controls', generated=len(records),
                        planned=16, route=route, source_id=row['id'])
                    print('source_generation', len(records), route, row['id'], flush=True)
        guard.check(full=True)
        if state_digest(code.state_dict()) != before_code or state_digest(original.state_dict()) != before_base:
            raise ValueError('No original or residual parameter may change during diagnosis')
        grouped = defaultdict(list)
        for row in records:
            grouped[(row['route'], row['role'])].append(row)
        summaries = [dict(route=route, role=role, source_rows=len(rows),
            first_reference_token_matches=sum(r['first_reference_token_match'] for r in rows),
            exact_complete_reference_matches=sum(r['exact_complete_reference_token_match'] for r in rows),
            terminal_EOS=sum(r['terminal_EOS'] for r in rows),
            exhausted_budget=sum(r['generated_tokens'] == 1024 and not r['terminal_EOS'] for r in rows),
            teacher_target_token_mean_NLL=sum(r['complete_reference_teacher_NLL'] * r['reference_tokens_including_EOS'] for r in rows)
                / sum(r['reference_tokens_including_EOS'] for r in rows)) for (route, role), rows in grouped.items()]
        result = dict(completed_at=datetime.now(timezone.utc).isoformat(), summaries=summaries,
            generations_sha256=file_hash(output), source_files_sha256=bound,
            original_base_state_unchanged=True, residual_state_unchanged=True,
            eight_original_source_sequences_directly_reencoded=True, optimizer_updates=0,
            functional_correctness_assessed=False, model_promoted=False, goal_achieved=False,
            scope='Eight short original source examples, four trained and four internal validation; sixteen raw greedy outputs from two shared own routes. Exact original reference reproduction and teacher NLL are diagnostics, not coding correctness or independent generalization evidence.')
        atomic_json(reports / 'results.json', result)
        status('source generation diagnostic complete; inspect full source-only evidence', summaries=summaries)
    except Exception as error:
        status('diagnostic stopped; preserve evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
