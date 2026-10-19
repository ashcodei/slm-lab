"""Read actual full984M control values, source masks and observed metric arithmetic.

No new neural inference or program execution. Neural likelihood numbers remain
the original experiment's observations. Adam values were not retained, so this
reader does not independently certify their clocks or durable recovery.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/function-termination-full-CPU-v1'
OUT = ROOT/'reports/function-termination-full-CPU-v1-readback'


def flatten(payload):
    return {group+'.'+name:value for group in ['network','heads'] for name,value in payload[group].items()
            if not (group=='network' and name=='lm_head.weight')}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():raise ValueError('Fresh CPU-only full-size source/value reader required')
    import torch
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    torch.set_num_threads(2)
    plan = json.loads((SOURCE/'plan.json').read_text());done = json.loads((SOURCE/'complete.json').read_text())
    if not terminal(plan['owner']) or not done['complete'] or (SOURCE/'failure.json').exists():
        raise ValueError('The exact complete matched full-size control must finish first')
    if (done['actual_unique_student_values'] != 984399690 or plan['starting_snapshot_actual_updates'] != 2048
            or plan['logical_source_updates_each_arm'] != 2 or plan['arms'] != dict(whole_CE=0.,CE_plus_true_EOS=.05)):
        raise ValueError('Exact fixed partial source and matched two-function control required')
    files = dict(done['source_files_sha256'])
    files[str(SOURCE/'complete.json')] = file_hash(SOURCE/'complete.json')
    files.update({str(p):file_hash(p) for p in python_sources(ROOT,[Path(__file__)])})
    pack = CompleteCodePack(ROOT/'data/documented-doctest-verified-v1','train','training')
    original = torch.load(plan['starting_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    initial = flatten(original['model'])
    if sum(t.numel() for t in initial.values()) != 984399690 or len(initial) != 461:
        raise ValueError('Every original unique text/image/word/auxiliary value must be counted')
    selected = plan['optimized_original_train_positions'];unoptimized = plan['unoptimized_diagnostic_original_train_positions']
    if len(selected) != 2 or len(unoptimized) != 2 or len(set(selected+unoptimized)) != 4:
        raise ValueError('Two optimized and two distinct unoptimized diagnostic train-pool references required')
    proofs = {};metrics = {}
    for arm,weight in plan['arms'].items():
        record = done['arms'][arm]
        model_path = Path(record['complete_native_model']);files[str(model_path)] = record['complete_native_model_sha256']
        if file_hash(model_path) != record['complete_native_model_sha256']:raise ValueError('Complete final native diagnostic bytes differ')
        payload = torch.load(model_path,map_location='cpu',weights_only=True,mmap=True)
        final = flatten(payload['complete_dense'])
        if (payload['completed_updates'] != 2 or payload['provenance']['EOS_auxiliary_weight'] != weight
                or payload['provenance']['optimized_original_positions'] != selected
                or payload['actual_counts']['total_unique'] != 984399690 or final.keys() != initial.keys()):
            raise ValueError('Exact complete standalone branch, source and diagnostic cursor required')
        changed = [];fixed = []
        for name,value in final.items():
            before = initial[name]
            if value.shape != before.shape or value.dtype != before.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Every exact finite original-shaped native tensor is required')
            language = name.startswith('network.') and not name.startswith(('network.vision.','network.value_head.'))
            same = torch.equal(value,before)
            if language:
                if same:raise ValueError('Every actual full dense language tensor should change in this observed control')
                changed.append(name)
            else:
                if not same:raise ValueError('Every inactive image/word/auxiliary value must remain exact')
                fixed.append(name)
        if len(changed) != 317 or len(fixed) != 144:raise ValueError('The actual full model partition must reconcile')
        if len(record['actual_update_records']) != 2:raise ValueError('Both original source updates required')
        for position,row in zip(selected,record['actual_update_records']):
            ids,labels = pack.sequence(position)
            if (row['original_source_position'] != position or row['complete_response_EOS_targets'] != sum(v!=-100 for v in labels[1:])
                    or row['complete_examples'] != 1 or row['EOS_auxiliary_weight'] != weight
                    or row['reference_neural_forwards'] != 0):
                raise ValueError('Every complete source target, true EOS and matched arm weight must reconstruct')
        for phase in ['before','after']:
            rows = record[phase]
            if [r['original_position'] for r in rows] != selected+unoptimized:raise ValueError('All four matched original observations required')
            for row in rows:
                ids,labels = pack.sequence(row['original_position'])
                raw = __import__('numpy').asarray([ids],dtype='<u2').tobytes()
                if (row['actual_original_input_ids_sha256'] != hashlib.sha256(raw).hexdigest()
                        or row['complete_context_tokens'] != len(ids)
                        or row['complete_response_EOS_targets'] != sum(v!=-100 for v in labels[1:])
                        or ids[-1] != 2 or labels[-1] != 2 or sum(v==2 for v in labels[1:]) != 1
                        or row['optimized_in_this_experiment'] != (row['original_position'] in selected)):
                    raise ValueError('Every complete native input/mask/EOS denominator and source label must rebuild')
        summary = {}
        for phase in ['before','after']:
            for optimized in [True,False]:
                rows = [r for r in record[phase] if r['optimized_in_this_experiment']==optimized]
                key = phase+('_optimized' if optimized else '_unoptimized_train_pool')
                summary[key] = dict(mean_true_EOS_NLL=sum(r['true_EOS_NLL'] for r in rows)/2,
                    mean_true_EOS_probability=sum(r['true_EOS_probability'] for r in rows)/2,
                    response_token_mean_NLL=sum(r['complete_response_EOS_CE_sum'] for r in rows)/sum(r['complete_response_EOS_targets'] for r in rows),
                    interior_argmax_EOS_count=sum(r['interior_response_argmax_EOS_count'] for r in rows))
        metrics[arm] = summary
        proofs[arm] = dict(actual_native_tensors=461,actual_changed_full_language_tensors=len(changed),
            actual_unchanged_vision_word_and_auxiliary_tensors=len(fixed),actual_unique_student_values=984399690,
            every_original_source_input_mask_and_true_EOS_rebuilt=True,
            native_model_sha256=record['complete_native_model_sha256'])
        del payload,final
    if done['arms']['whole_CE']['before'] != done['arms']['CE_plus_true_EOS']['before']:
        raise ValueError('Every actual initial source observation must match')
    guard = FileBindings(files);guard.check(full=True);OUT.mkdir()
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        actual_full_native_value_proofs=proofs,arithmetic_from_original_observed_neural_likelihoods=metrics,
        actual_initial_observations_match=True,source_files_sha256=files,
        original_gold_function_endings_already_high_confidence_EOS_on_these_four_sources=True,
        extra_EOS_weight_improves_observed_terminal_likelihood_but_not_a_proved_free_generation_fix=True,
        both_arms_worsen_unoptimized_train_pool_token_NLL_in_this_tiny_control=True,
        unoptimized_sources_are_not_new_held_or_all_history_unseen_benchmarks=True,
        independent_new_neural_likelihoods_not_computed=True,Adam_values_not_retained_no_independent_durable_recovery_claim=True,
        no_generated_correctness_or_efficiency_gain_measured=True,new_model_updates=0,
        original_GPU_jobs_and_data_unchanged=True,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_full_native_value_proofs=proofs,
        observed_likelihood_arithmetic=metrics,independent_neural_forwards=0)),flush=True)


if __name__ == '__main__':main()
