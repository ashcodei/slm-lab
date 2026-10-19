"""Scope/dependency gate controls only; fixture reports are never real evidence."""
import copy
import json
from pathlib import Path

import pytest

from slm_lab import joint_visual_runtime_contract_v1 as contract


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup(root, monkeypatch):
    owners = dict(pipeline_owner=dict(pid=1, ticks='100'), finalizer_owner=dict(pid=2, ticks='200'),
        source_owner=dict(pid=3, ticks='300'))
    monkeypatch.setattr(contract, 'terminal', lambda owner: True)
    for folder, owner in [(contract.PIPELINE, owners['pipeline_owner']), (contract.FINALIZER, owners['finalizer_owner']),
            (contract.SOURCE, owners['source_owner'])]:
        write(root, folder + '/plan.json', dict(owner=owner))
    checkpoint = root / 'fake-native-for-control-only.pt'
    checkpoint.write_bytes(b'not-a-model; gate controls cannot replace actual native loading or GPU capacity')
    digest = contract.sha(checkpoint)
    final = dict(complete=True, completed_updates=102, actual_counts=dict(total_unique=984399690),
        checkpoint=str(checkpoint), checkpoint_sha256=digest,
        all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact=True,
        real_word_logits_exact_through_fold_and_cold_native=True,
        old_checkpoint_actor_optimizer_or_reference_not_needed_for_inference=True,
        fixed_machine_DPO_export_not_quality_winner=True, all_original_machine_source_updates_complete=True)
    write(root, contract.FINALIZER + '/complete.json', final)
    write(root, contract.PIPELINE + '/plan.json', dict(owner=owners['pipeline_owner'], standalone=dict(checkpoint=str(checkpoint),
        checkpoint_sha256=digest, receipt_sha256=contract.sha(root / contract.FINALIZER / 'complete.json'))))
    write(root, contract.PIPELINE_QUEUE, dict(owner=owners['pipeline_owner'], expected_student_values=984399690,
        **contract.SCOPES, all_original_case_populations_budgets_and_source_exposure_disclosures_preserved=True))
    write(root, contract.FINALIZER_QUEUE, dict(owner=owners['finalizer_owner']))
    write(root, contract.PIPELINE + '/complete.json', dict(complete=True, actual_unique_student_values=984399690,
        **{key: True for key in contract.COMPLETION}))
    write(root, contract.PIPELINE + '/restored.json', dict(all_registered_point_actors_terminal=True))
    write(root, contract.PIPELINE + '/children.json', [dict(pid=4, ticks='400')])
    for name, count in [('full-QA.json', 16961), ('full-AI2D.json', 3088), ('full-original-MBPP.json', 600),
            ('held-preferences.json', 4642)]:
        write(root, contract.PIPELINE + '/standalone/' + name, dict(results=[dict(fixture=True)] * count))
    word = dict(results=[dict(source_position=p, conditions={c: {} for c in ['proper', 'blank', 'wrong']}) for p in range(3429)],
        generation_budget=64, complete_held_source_words=True, held_source_gradients=False,
        all_original_global_wrong_donors_preserved=True)
    write(root, contract.PIPELINE + '/standalone/all-held-words.json', word)
    for suite in ['code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words']:
        for rank in range(4):
            write(root, contract.PIPELINE + f'/standalone/assessment-{suite}-{rank}-complete.json', dict(physical_GPU=rank,
                actual_student_values=984399690, all_loaded_values_unchanged=True,
                only_standalone_weights_and_tokenizer_for_neural_inference=True, checkpoint_sha256=digest, model_updates=0))
    admission = dict(owner=owners['source_owner'], training_admitted=True, admitted_role='train', complete_source_views=302627,
        complete_physical_paths=195121, complete_global_steps=4733, complete_rank_calls=18932, complete_context_tokens=17681556,
        complete_response_EOS_targets=4669243, whole_word_views=91585, whole_CTC_UTF8_bytes=644063,
        four_complete_original_negative_targets=80, runtime_or_kernel_or_model_admission_not_granted=True,
        historical_failed_word_and_prefix_numeric_controls_not_waived=True)
    write(root, contract.ADMISSION, admission)
    write(root, contract.SOURCE + '/complete.json', dict(complete=True, passed=True, data_only_training_admitted=True,
        admission_sha256=contract.sha(root / contract.ADMISSION)))
    return owners


def change(root, path, edit):
    value = contract.read(root / path)
    edit(value)
    write(root, path, value)


def test_complete_scope_gate_preserves_original_owners_and_does_not_claim_runtime_capacity(tmp_path, monkeypatch):
    owners = setup(tmp_path, monkeypatch)
    assert contract.ready(tmp_path, **owners)
    assert not (tmp_path / 'GPU-capacity-complete.json').exists()


def test_a_verified_live_original_owner_is_waited_on_without_reading_future_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, 'terminal', lambda owner: False)
    assert contract.ready(tmp_path, pipeline_owner=dict(pid=1, ticks='100'),
        finalizer_owner=dict(pid=2, ticks='200'), source_owner=dict(pid=3, ticks='300')) is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('case', ['partial_QA', 'lost_word_condition', 'partial_words', 'wrong_scope', 'unrestored',
    'live_registered_worker', 'changed_checkpoint', 'failed_predecessor', 'different_birth', 'tiny_counts',
    'lost_negative_targets', 'missing_actual_worker_proof'])
def test_no_partial_scope_or_wrong_native_source_or_unfinished_worker_can_unlock_continuation(case, tmp_path, monkeypatch):
    owners = setup(tmp_path, monkeypatch)
    if case == 'partial_QA':
        change(tmp_path, contract.PIPELINE + '/standalone/full-QA.json', lambda a: a['results'].pop())
    elif case == 'lost_word_condition':
        change(tmp_path, contract.PIPELINE + '/standalone/all-held-words.json', lambda a: a['results'][0]['conditions'].pop('wrong'))
    elif case == 'partial_words':
        change(tmp_path, contract.PIPELINE + '/standalone/all-held-words.json', lambda a: a['results'].pop())
    elif case == 'wrong_scope':
        change(tmp_path, contract.PIPELINE_QUEUE, lambda a: a.update(full_MBPP_problems=24))
    elif case == 'unrestored':
        change(tmp_path, contract.PIPELINE + '/restored.json', lambda a: a.update(all_registered_point_actors_terminal=False))
    elif case == 'live_registered_worker':
        monkeypatch.setattr(contract, 'terminal', lambda owner: owner['pid'] != 4)
    elif case == 'changed_checkpoint':
        (tmp_path / 'fake-native-for-control-only.pt').write_bytes(b'changed')
    elif case == 'failed_predecessor':
        write(tmp_path, contract.SOURCE + '/failure.json', dict(error='deliberate control'))
    elif case == 'different_birth':
        change(tmp_path, contract.PIPELINE_QUEUE, lambda a: a.update(owner=dict(pid=1, ticks='101')))
    elif case == 'tiny_counts':
        change(tmp_path, contract.PIPELINE + '/complete.json', lambda a: a.update(actual_unique_student_values=70338))
    elif case == 'lost_negative_targets':
        change(tmp_path, contract.ADMISSION, lambda a: a.update(four_complete_original_negative_targets=0))
        change(tmp_path, contract.SOURCE + '/complete.json', lambda a: a.update(admission_sha256=contract.sha(tmp_path / contract.ADMISSION)))
    else:
        change(tmp_path, contract.PIPELINE + '/standalone/assessment-words-3-complete.json', lambda a: a.update(model_updates=1))
    with pytest.raises(ValueError):
        contract.ready(tmp_path, **owners)
