"""CPU readback of saved precision vectors, full source tokens, updates and state."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(1)
    precision = ROOT / 'reports/code-residual-inference-precision-v1'
    learning = ROOT / 'reports/four-source-code-learning-probe-v1'
    output = ROOT / 'reports/source-learning-diagnostics-v1-independent-readback.json'
    if output.exists():
        raise ValueError('Preserve any previous independent readback')
    a, b = read(precision / 'results.json'), read(learning / 'results.json')
    bindings = {**a['source_files_sha256'], **b['source_files_sha256']}
    assert all(file_hash(Path(p)) == h for p, h in bindings.items())
    assert file_hash(precision / 'records.json') == a['records_sha256']
    assert file_hash(precision / 'full-next-token-logits.pt') == a['full_logits_sha256']
    assert file_hash(learning / 'plan.json') == b['plan_sha256']
    assert file_hash(learning / 'generations.json') == b['generation_report_sha256']
    plan = read(learning / 'plan.json')
    controls = {r['id']: r for r in plan['source_rows']}
    assert len(controls) == 8
    assert not ({r['group_id'] for r in controls.values() if r['role'] == 'train'} &
                {r['group_id'] for r in controls.values() if r['role'] == 'validation'})
    tokenizer = Tokenizer.from_file(str(ROOT / 'checkpoints/code-residual-lr-pilot-v1/tokenizer.json'))
    originals, references = {}, {}
    with gzip.open(ROOT / 'data/codenet-durable-pack-v2/original-rows.jsonl.gz', 'rt') as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            if row['id'] not in controls:
                continue
            c = controls[row['id']]
            assert row['split'] == c['role'] and row['group_id'] == c['group_id']
            assert position == row['token_pack_index'] == c['global_position']
            framed_prompt = (row['prompt'] if row['prompt'].rstrip().endswith('Assistant:')
                             else 'User: ' + row['prompt'] + '\nAssistant:')
            prompt = tokenizer.encode(framed_prompt).ids
            reference = tokenizer.encode(row['response']).ids + [2]
            ids = [1] + prompt + reference
            digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()
            assert digest == row['complete_input_ids_sha256'] == c['full_input_ids_sha256']
            assert len(ids) == c['context_tokens'] and len(reference) == c['target_tokens']
            originals[row['id']], references[row['id']] = row, reference
    assert set(originals) == set(controls)

    def agreement(record, source):
        generated, reference = record['generated_token_ids'], references[source]
        assert 0 < len(generated) <= record['generation_budget']
        assert all(type(v) is int and 0 <= v < 32768 for v in generated)
        assert 2 not in generated[:-1]
        prefix = 0
        for x, y in zip(generated, reference):
            if x != y:
                break
            prefix += 1
        expected = dict(first_reference_token_match=generated[0] == reference[0],
            matching_reference_prefix_tokens=prefix,
            exact_complete_reference_token_match=generated == reference,
            terminal_EOS=generated[-1] == 2,
            unique_contiguous_token_4grams=len({tuple(generated[i:i+4]) for i in range(len(generated)-3)}))
        assert all(record[k] == v for k, v in expected.items())
        return tokenizer.decode(generated[:-1] if generated[-1] == 2 else generated,
                                skip_special_tokens=False)

    records = read(precision / 'records.json')['records']
    vectors = torch.load(precision / 'full-next-token-logits.pt', map_location='cpu', weights_only=True)
    assert len(vectors) == len(records) == 24
    tied = []
    for r, v in zip(records, vectors):
        assert (r['source_id'], r['mode']) == (v['source_id'], v['mode'])
        assert r['generation_budget'] == 64 and agreement(r, r['source_id']) == r['raw_text']
        for name in ['full', 'cached', 'repeated']:
            x = v[name]
            assert x.shape == (32768,) and bool(torch.isfinite(x).all())
            gold = references[r['source_id']][0]
            d = r[name]
            assert d['argmax_token_id'] == int(x.argmax())
            assert d['gold_token_id'] == gold
            assert d['gold_strict_greater_rank'] == int((x > x[gold]).sum()) + 1
            assert d['gold_tied_values'] == int((x == x[gold]).sum())
            assert d['maximum_tied_values'] == int((x == x.max()).sum())
        assert r['cached_full_max_abs_error'] == float((v['full'] - v['cached']).abs().max())
        assert r['repeated_cached_max_abs_error'] == float((v['repeated'] - v['cached']).abs().max())
        assert r['cached_argmax_matches_generated1'] == (int(v['cached'].argmax()) == r['generated1_token'])
        assert r['full_argmax_matches_uncached1'] == (int(v['full'].argmax()) == r['uncached1_token'])
        if r['cached']['maximum_tied_values'] > 1:
            tied.append(dict(mode=r['mode'], source=r['source_id'], **r['cached']))
    for summary in a['summaries']:
        rows = [r for r in records if r['mode'] == summary['mode']]
        expected = dict(mode=summary['mode'], controls=len(rows),
            maximum_cached_full_logit_error=max(r['cached_full_max_abs_error'] for r in rows),
            maximum_repeated_prefill_logit_error=max(r['repeated_cached_max_abs_error'] for r in rows),
            cached_argmax_generate1_agreements=sum(r['cached_argmax_matches_generated1'] for r in rows),
            full_argmax_uncached1_agreements=sum(r['full_argmax_matches_uncached1'] for r in rows),
            maximum_logit_ties_controls=sum(r['cached']['maximum_tied_values'] > 1 for r in rows),
            first_reference_token_matches=sum(r['first_reference_token_match'] for r in rows),
            terminal_EOS_within64=sum(r['terminal_EOS'] for r in rows))
        assert expected == summary
    assert a['original_state_sha256'] == a['final_state_sha256']

    checkpoint = Path(b['checkpoint'])
    assert file_hash(checkpoint) == b['checkpoint_sha256'] and checkpoint.stat().st_size == b['checkpoint_bytes']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert state['stage'] == 'code_residual_four_source_learning_probe_v1'
    assert state['steps'] == state['planned_steps'] == b['actual_updates'] == 256
    assert state['run_plan_sha256'] == b['plan_sha256']
    assert state['fresh_optimizer'] and state['memorization_diagnostic_only']
    assert state['base_checkpoint_sha256'] == plan['original_base_checkpoint_sha256']
    assert state['starting_policy_checkpoint_sha256'] == plan['starting_policy_sha256']
    expected_names = [f'{i}.{name}.weight' for i in range(24) for name in ['down', 'up']]
    assert state['residual_parameter_names'] == expected_names
    assert set(state['residuals']) == set(expected_names)
    assert state['model_counts'] == b['parameter_counts'] == dict(total_unique=857007616,
        frozen_base=706012672, trainable_code=150994944)
    assert sum(t.numel() for t in state['residuals'].values()) == 150994944
    assert state_digest(state['residuals']) == b['final_residual_state_sha256']
    assert b['initial_residual_state_sha256'] != b['final_residual_state_sha256']
    opt = state['optimizer']
    assert opt['param_groups'][0]['params'] == list(range(48))
    assert set(opt['state']) == set(range(48))
    for i, name in enumerate(expected_names):
        tensor, moments = state['residuals'][name], opt['state'][i]
        assert tensor.shape == ((2048,1536) if '.down.' in name else (1536,2048))
        assert bool(torch.isfinite(tensor).all()) and tensor.dtype == torch.float32
        assert set(moments) == {'step', 'exp_avg', 'exp_avg_sq'} and float(moments['step']) == 256
        for key in ['exp_avg', 'exp_avg_sq']:
            assert moments[key].shape == tensor.shape and bool(torch.isfinite(moments[key]).all())
        assert bool((moments['exp_avg_sq'] >= 0).all())
    rng = state['rng']
    assert set(rng) == {'python', 'numpy', 'torch', 'cuda'}
    random.Random().setstate(rng['python'])
    n = rng['numpy']
    np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32),
                                     n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.Generator().set_state(rng['torch'])
    assert rng['cuda'].dtype == torch.uint8 and rng['cuda'].ndim == 1 and rng['cuda'].numel() > 0
    metrics = state['metrics']
    assert [m['step'] for m in metrics] == list(range(1,257))
    assert all(m['target_tokens'] == 674 and np.isfinite(m['mean_NLL_before_update']) for m in metrics)
    assert sum(m['target_tokens'] for m in metrics) == b['response_target_exposures'] == 172544
    base = torch.load(plan['original_base_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    assert state_digest(base['model']) == state['frozen_backbone_state_sha256'] == b['original_base_state_sha256'] == b['final_base_state_sha256']
    generated = read(learning / 'generations.json')['records']
    assert len(generated) == 16 and len({(r['step'],r['id']) for r in generated}) == 16
    for r in generated:
        assert r['generation_budget'] == 1024 and r['role'] == controls[r['id']]['role']
        raw = agreement(r, r['id'])
        assert raw == r['raw_program_text'] and hashlib.sha256(raw.encode()).hexdigest() == r['raw_program_sha256']
    for observation in b['observations']:
        for role, key in [('train','train_NLL'), ('validation','validation_NLL')]:
            scores = observation[key]['rows']
            assert {r['id'] for r in scores} == {k for k,v in controls.items() if v['role']==role}
            mean = sum(r['targets']*r['mean_NLL'] for r in scores)/sum(r['targets'] for r in scores)
            assert mean == observation[key]['token_mean_NLL']
        for role in ['train','validation']:
            key = role + '_generation'
            if key not in observation:
                continue
            rows = [r for r in generated if r['step']==observation['step'] and r['role']==role]
            assert observation[key] == dict(sources=len(rows),
                exact_reference_matches=sum(r['exact_complete_reference_token_match'] for r in rows),
                first_reference_token_matches=sum(r['first_reference_token_match'] for r in rows),
                terminal_EOS=sum(r['terminal_EOS'] for r in rows))
    atomic_json(output, dict(verified_at=datetime.now(timezone.utc).isoformat(),
        source=str(Path(__file__)), source_sha256=file_hash(Path(__file__)),
        source_bindings_rehashed=len(bindings), saved_precision_vectors_rederived=24,
        precision_summaries=a['summaries'], precision_maximum_tie_cases=tied,
        complete_original_sources_directly_reencoded=8, source_generations_rederived=16,
        complete_checkpoint_sha256=b['checkpoint_sha256'], complete_checkpoint_bytes=b['checkpoint_bytes'],
        all48_residuals_and_Adam_steps_verified=True, RNG_inventory_verified_without_GPU_restore=True,
        original_base_full_state_rederived=True, update_targets_rederived=172544,
        final_observation=b['observations'][-1], candidate_promoted=False, goal_achieved=False,
    scope='Saved-artifact readback; not independent model rerun or new functional/generalization evidence. Four training references are memorized; validation loss worsens.'))
    print(json.dumps(dict(verified=True, updates=256, training_exact='4/4', validation_exact='0/4',
        validation_EOS='4/4', report=str(output))))


if __name__ == '__main__':
    main()
