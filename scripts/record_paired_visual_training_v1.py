"""Independent complete-state readback; distinguish fixture proof from real work."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.finalized_visual_pairs_v1 import FinalizedVisualPairs
from slm_lab.finalized_visual_replay_v1 import FinalizedVisualReplay, VISUAL_FAMILIES
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.paired_visual_training_v1 import FORMAT

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports'
    destination = reports / 'paired-visual-training-v1-CPU'
    result_path = reports / 'paired-visual-training-v1-readiness.json'
    if destination.exists() or result_path.exists():
        raise ValueError('Preserve existing retained evidence; inspect before recording again')
    source = Path('/tmp/pytest-of-jovyan/pytest-107/test_four_worker_complete_visu0')
    logs = [reports / 'paired-visual-training-v1-release-tests.log', reports / 'finalized-visual-replay-v1-focused-tests.log']
    for log, n in zip(logs, [2, 1]):
        if f'{n} passed' not in log.read_text() or 'FAILED' in log.read_text():
            raise ValueError('Current focused checks required')
    ranks = [json.loads((source / f'rank-{r}.json').read_text()) for r in range(4)]
    if any(r['rank'] != i or r['complete_reference_updates'] != 16 or r['checkpoint_step'] != 7
            or r['exact_resumed_updates'] != 9 or not r['all_eight_visual_families_complete']
            or not r['full_weights_Adam_each_rank_RNG_metrics_cursors_match']
            or not r['one_rank_frozen_weight_change_blocks_snapshot'] for i, r in enumerate(ranks)):
        raise ValueError('Complete actual four-worker recovery evidence required')
    states = {n: torch.load(source / (n + '.pt'), weights_only=True, map_location='cpu')
        for n in ['resume', 'final', 'recovered']}
    if fingerprint(states['final']) != fingerprint(states['recovered']):
        raise ValueError('Full neural/Adam/worker-RNG/metrics/cursors recovery differs')
    for name, state in states.items():
        expected = 7 if name == 'resume' else 16
        if (state['format'] != FORMAT or state['steps'] != expected or state['world_size'] != 4
                or state['state_sha256'] != fingerprint({k: v for k, v in state.items() if k != 'state_sha256'})
                or len(state['metrics']) != expected or state['sampler_state']['updates'] != expected):
            raise ValueError('Complete saved visual state differs')
        for moment in state['optimizer']['state'].values():
            if float(moment['step']) != expected or not all(bool(torch.isfinite(moment[n]).all()) for n in ['exp_avg', 'exp_avg_sq']):
                raise ValueError('Complete finite step-matched vision Adam required')
        for row in state['metrics']:
            if not math.isclose(row['summed_objective'], row['paired']['summed_objective'] + row['earlier_visual_replay']['CE_summed_objective'], abs_tol=1e-12):
                raise ValueError('Paired and earlier-visual objectives must reconcile')
    state = states['final']; config = ModelConfig(**state['config'])
    torch.manual_seed(49216)
    initial = TinyMultimodalLM(config)
    learned = TinyMultimodalLM(config); learned.load_state_dict(state['model'], strict=True)
    for name, tensor in initial.state_dict().items():
        if not name.startswith('vision.') and not torch.equal(tensor, state['model'][name]):
            raise ValueError('Original frozen language or critic changed')
    changed = [n for n, t in initial.vision.state_dict().items() if not torch.equal(t, learned.vision.state_dict()[n])]
    if not changed or len({fingerprint(rng) for rng in state['worker_rng_states']}) != 4:
        raise ValueError('Actual vision updates and distinct exact rank RNG states required')
    initial.eval(); learned.eval()
    ids = torch.tensor([[1, 5, 11, 2]])
    with torch.no_grad():
        if not torch.equal(initial(ids)['logits'], learned(ids)['logits']):
            raise ValueError('Text-only logits changed')
    # Independent permutations recover the complete fixture exposure inventory.
    reached = {n: set() for n in VISUAL_FAMILIES}
    for row in state['metrics']:
        name, occurrence = row['replay_family'], row['replay_occurrence']
        number = state['identity']['populations'].index((name, 5))
        for draw in range(occurrence * 4, (occurrence + 1) * 4):
            epoch, cursor = divmod(draw, 5)
            order = list(range(5)); random.Random(438 + 1000 + number + epoch).shuffle(order)
            reached[name].add(order[cursor])
    if any(indices != set(range(5)) for indices in reached.values()):
        raise ValueError('Every fixture visual source row must actually run')

    earlier = json.loads((reports / 'paired-visual-grounding-v1-readiness.json').read_text())
    for path, digest in earlier['files_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('Prior original-photo/706M controls or gradient evidence changed')
    actual_token = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(actual_token)); actual_config = ModelConfig.load(ROOT / 'configs/model.json')
    gates = {}
    for name, reader in [('paired_photos', FinalizedVisualPairs), ('earlier_visual', FinalizedVisualReplay)]:
        try:
            opened = reader(ROOT, tokenizer, actual_config, 448)
            gates[name] = dict(opened=True, source_contract_sha256=opened.contract_sha256)
        except ValueError as error:
            if not any(t in str(error) for t in ['Finalized original real-photo pair manifest', 'Finalized complete30-family source bundle']):
                raise
            gates[name] = dict(opened=False, rejected_unfinalized_source=True, reason=str(error))

    inventory = {}; bindings = {}
    for relative in ['codenet-replay-v1', 'codenet-image-replay-v1']:
        folder = ROOT / 'data' / relative
        manifest = folder / 'manifest.json'; m = json.loads(manifest.read_text()); index = folder / 'index.npy'
        if file_hash(index) != m['artifact_files_sha256']['index.npy']:
            raise ValueError('Actual existing visual index differs')
        bindings[str(manifest)] = file_hash(manifest); bindings[str(index)] = file_hash(index)
        rows = np.load(index, allow_pickle=False, mmap_mode='r')
        for i, name in enumerate(m['family_names']):
            if name in VISUAL_FAMILIES:
                selected = rows[rows['family'] == i]
                if len(selected) != m['family_counts'][name]:
                    raise ValueError('Actual admitted visual population differs')
                inventory[name] = dict(original_logical_records=len(selected), max_context_tokens=int(selected['context_tokens'].max()),
                    proper_target_tokens=int(selected['target_tokens'].astype(np.int64).sum()),
                    inventory_only_not_new_source_admission=True)
    if set(inventory) != set(VISUAL_FAMILIES):
        raise ValueError('All existing eight visual inventories required')

    def process(pid):
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            raise ValueError('Expected original actor is terminal')
        return dict(pid=pid, parent_pid=int(fields[1]), start_ticks=fields[19],
            argv=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    actors = {n: process(pid) for n, pid in [('collector', 130149), ('controller', 2754743),
        ('human_feedback_waiter', 130151), ('paired_preparer', 2755925), ('full_source_preparer', 2756113)]}
    app_rows = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout.splitlines()
    workers = [dict(**process(int(row.split(',')[0])), GPU_record=row) for row in app_rows]
    if len(workers) != 4 or any(w['parent_pid'] != 2841909 or 'slm_lab.training pretrain' not in w['argv'] for w in workers):
        raise ValueError('Reinspect actual original four-GPU training ownership')
    metrics = [json.loads(line) for line in (reports / 'curriculum-disk-cleanup-resumption-v1/joint.log').read_text().splitlines()
        if line.startswith('{"step"')]
    collector_rows = list((ROOT / 'data/vqa-real-pairs-v1/images.collected.jsonl').open())
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('rendezvous', '*.lock'))
    files = [ROOT / 'src/slm_lab' / (n + '.py') for n in ['paired_visual_training_v1', 'finalized_visual_replay_v1']]
    files += [ROOT / 'tests' / (n + '.py') for n in ['test_paired_visual_training_v1', 'test_finalized_visual_replay_v1']]
    files += [Path(__file__).resolve(), reports / 'paired-visual-grounding-v1-readiness.json', *logs]
    files += [p for p in destination.iterdir() if p.is_file()]
    bindings.update({str(path): file_hash(path) for path in files})
    result = dict(format='own-paired-visual-training-readiness-v1', observed_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification=dict(classification='progress', reason=earlier['current_turn_classification']['reason']),
        current_turn_classification=dict(classification='progress', reason='Implemented original vision-only paired CE/ranking plus complete eight-family visual replay and complete-state recovery; three new focused checks pass.'),
        implementation=dict(full_training_kernel_implemented=True, complete_model_checkpoint_with_Adam_RNG_metrics_and_cursors=True,
            source_factory_requires_finalized_original_full_bundle=True, production_launcher_integrated=False,
            actual_full706M_grounding_updates=0, actual_full706M_GPU_capacity_proven=False, added_parameters=0),
        checks=dict(new_focused_checks=3, earlier_separate_gradient_and_reader_checks=8, ranks=ranks,
            independent_readback=dict(full_recovered_state_matches=True, original_language_critic_and_text_logits_identical=True,
                changed_vision_tensors=changed, all_earlier_fixture_visual_rows_reached=True, model_counts=initial.parameter_counts())),
        actual_existing_visual_inventory=inventory, actual_existing_logical_visual_records=sum(r['original_logical_records'] for r in inventory.values()),
        prospective_replay_only_minimum_updates={str(acc): sum(math.ceil(r['original_logical_records'] / (4 * acc)) for r in inventory.values()) for acc in [8, 16, 32]},
        prospective_layout_scope='Inventory-derived minima only, not a run plan, throughput estimate, finalized paired source or capacity proof.',
        real_source_factory_gates=gates, actual_original_actors=actors, actual_original_GPU_workers=workers,
        actual_original_latest_metric=metrics[-1], collector_current_committed_rows=len(collector_rows),
        prior_actual706M_controls=earlier['actual_own706M_CPU_diagnostic'],
        remaining_requirements=['Finalize original paired/full30 sources and exact completed own base; retain original continuation and genuine-human feedback.',
            'Integrate full-model capacity check, production lifecycle and matched original native paired/caption/OCR/object/domain/coding assessments before promotion.',
            'Run actual full-model training and demonstrate useful multimodal, code correctness/efficiency and broad knowledge gains; high-end parity remains unproven.'],
        candidate_promoted=False, new_capability_gain_measured=False, goal_achieved=False, files_sha256=bindings)
    atomic_json(result_path, result)
    print(json.dumps(dict(report=str(result_path), new_checks=3, model_counts=initial.parameter_counts(),
        actual_existing_visual_records=result['actual_existing_logical_visual_records'],
        real_source_factory_gates=gates, actual_original_step=metrics[-1]['step'], goal_achieved=False), indent=2))


if __name__ == '__main__':
    main()
