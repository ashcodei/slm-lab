"""Independent retained extension/Adam/source readback; no new model forward."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_residual_checkpoint_v1 import _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_aligned_assistant_v1 import FORMAT, GROUPS
from slm_lab.own_alignment_checkpoint_v1 import FORMAT as RESUME_FORMAT

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-aligned-assistant-full-CPU-v1'


def row_at(path, position):
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index == position:
                return json.loads(line)
    raise ValueError('Actual original source position missing')


def sequence(tokenizer, row, field):
    prompt = row['prompt']
    prompt = prompt if prompt.rstrip().endswith('Assistant:') else 'User: '+prompt+'\nAssistant:'
    prefix = [1, *tokenizer.encode(prompt).ids]
    response = [*tokenizer.encode(row[field]).ids, 2]
    if len(prefix)+len(response) > 4096:
        raise ValueError('Whole original source must fit without truncation')
    return dict(input_ids=[prefix+response], labels=[[-100]*len(prefix)+response],
                attention_mask=[[True]*(len(prefix)+len(response))])


def main():
    destination = REPORTS / 'independent-extension-and-source-readback.json'
    if destination.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden independent CPU readback required')
    torch.set_num_threads(2)
    complete = json.loads((REPORTS / 'complete.json').read_text())
    plan = json.loads((REPORTS / 'plan.json').read_text())
    files = {**plan['source_files_sha256'], str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json'),
        str(REPORTS/'complete.json'):file_hash(REPORTS/'complete.json')}
    for name, evidence in complete['artifacts'].items():
        path = REPORTS/name
        if path.stat().st_size != evidence['bytes']:
            raise ValueError('Every actual retained artifact must match its full byte count')
        files[str(path)] = evidence['sha256']
    guard = FileBindings(files)
    initial = torch.load(REPORTS/'initial-extensions.pt', map_location='cpu', weights_only=True, mmap=True)
    final = torch.load(REPORTS/'disposable-trained-extensions.pt', map_location='cpu', weights_only=True, mmap=True)
    resume = torch.load(REPORTS/'disposable-resume.pt', map_location='cpu', weights_only=True, mmap=True)
    if (initial['format'] != FORMAT or final['format'] != FORMAT or resume['format'] != RESUME_FORMAT
            or initial['actual_counts'] != complete['actual_counts'] or final['actual_counts'] != complete['actual_counts']
            or resume['model']['actual_counts'] != complete['actual_counts']
            or sum(t.numel() for t in final['extensions'].values()) != 13409474
            or complete['actual_counts']['total_unique'] != 999860418
            or set(final['extensions']) != set(initial['extensions'])
            or resume['steps'] != 1 or resume['planned_steps'] != 2 or resume['metrics'] != complete['updates'][:1]
            or set(resume['model']['extensions']) != set(final['extensions'])
            or final['sources'] != initial['sources'] or resume['model']['sources'] != final['sources']
            or final['sources']['run_plan_sha256'] != file_hash(REPORTS/'plan.json')):
        raise ValueError('Exact complete source-bound initial/final/resume inventory and actual count required')
    for name, tensor in final['extensions'].items():
        if (any(word in name for word in ('embedding', 'original', 'base.', 'lm_head'))
                or any(t.shape != tensor.shape or t.dtype != tensor.dtype or not bool(torch.isfinite(t).all())
                       for t in (tensor, initial['extensions'][name], resume['model']['extensions'][name]))):
            raise ValueError('Only complete finite new tensors may appear in an extension export')
        if name.endswith('.up') and not bool((initial['extensions'][name] == 0).all()):
            raise ValueError('Every original added matrix must begin with an exact zero up-projection')
    names = [name for group in resume['groups'] for prefix in GROUPS[group]
             for name in final['extensions'] if name.startswith(prefix+'.')]
    adam = resume['optimizer']
    if (names != resume['optimized_names'] or adam['param_groups'][0]['params'] != list(range(len(names)))
            or set(adam['state']) != set(range(len(names))) or len(names) != complete['complete_Adam_states']
            or any(name.startswith('critic.') for name in names)):
        raise ValueError('Every and only selected new parameter needs its exact ordered Adam state')
    for index, name in enumerate(names):
        value = adam['state'][index]; tensor = final['extensions'][name]
        if (set(value) != {'step','exp_avg','exp_avg_sq'} or float(value['step']) != 1
                or any(value[k].shape != tensor.shape or value[k].dtype != tensor.dtype or not bool(torch.isfinite(value[k]).all())
                       for k in ('exp_avg','exp_avg_sq')) or bool((value['exp_avg_sq']<0).any())):
            raise ValueError('Complete exact finite actual step-one Adam moments required')
    _validate_rng(resume['rng'], 'cpu')
    if not torch.equal(initial['extensions']['critic.weight'], final['extensions']['critic.weight']):
        raise ValueError('Unused own PPO critic must remain unchanged')
    old_plan = json.loads((ROOT/'reports/own-onpolicy-pilot-v2/plan.json').read_text())
    tokenizer = Tokenizer.from_file(old_plan['tokenizer'])
    paths = dict(human='data/human-dialogue-v1/train.jsonl', ranking='data/human-preference-curriculum-v1/preferences.train.jsonl',
                 word='data/expanded-scene-words-v1/train.jsonl')
    rows = {name:row_at(ROOT/path,plan['actual_sources'][name]['position']) for name,path in paths.items()}
    if any(rows[name]['id'] != plan['actual_sources'][name]['id'] for name in rows):
        raise ValueError('Original train-source row identities required')
    expected = dict(human=sequence(tokenizer, rows['human'], 'response'),
        ranking={name:sequence(tokenizer, rows['ranking'], name) for name in ('chosen','rejected')},
        word=sequence(tokenizer, rows['word'], 'response'))
    if (expected != complete['source_sequences'] or rows['human']['synthetic'] is not False
            or rows['ranking']['feedback_source'] != 'human'
            or rows['ranking']['response_origin'] != complete['ranking_response_origin']
            or rows['word']['response'] != complete['whole_word_transcript']
            or file_hash(rows['word']['image']) != rows['word']['stored_image_sha256']
            or any(complete.get(k) is not True for k in
                ('actual_next_update_every_extension_Adam_and_RNG_recovery_exact',
                 'existing_original_and_code_and_grounding_bytes_versions_and_native_route_logits_exact',
                 'actual_all_new_only_tensor_cold_reload_and_aligned_and_word_routes_exact'))):
        raise ValueError('Every complete actual source framing, label/mask and original provenance qualification required')
    guard.check(full=True)
    report = dict(completed_at=datetime.now(timezone.utc).isoformat(), reader_sha256=file_hash(Path(__file__).resolve()),
        plan_sha256=file_hash(REPORTS/'plan.json'), full_model_probe_sha256=file_hash(REPORTS/'complete.json'),
        every_bound_source_and_artifact_rehashed=True, source_files=len(files), actual_counts=complete['actual_counts'],
        original_backbones_reloaded_by_this_reader=0, fresh_neural_forward_performed_by_this_reader=False,
        full_model_actual_forward_gradient_recovery_evidence='complete.json (source-bound actual CPU process)',
        all_actual_new_tensors_finite_and_complete=True, original_weights_absent_from_new_only_files=True,
        all_actual_step_one_Adam_states_reconciled=len(names), complete_own_BPE_BOS_response_EOS_masks_reconstructed=True,
        human_source_id=rows['human']['id'], human_ranked_pair_id=rows['ranking']['id'], human_word_source_id=rows['word']['id'],
        human_ranked_response_origin=rows['ranking']['response_origin'], source_word_mirror_caveat_retained=True,
        CUDA_initialized=False, production_training_not_performed=True, capability_improvement_not_established=True,
        candidate_promoted=False, goal_achieved=False)
    atomic_json(destination, report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
