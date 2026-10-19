"""Matched true-EOS auxiliary versus whole-function CE on our fixed984M source.

Two already-admitted complete references are optimized once per arm; two more
admitted train-pool references are unoptimized diagnostic controls. These are
not independent benchmark scores. This fixed partial snapshot experiment is
separate from the live full epoch and makes no production/promotion claim.
"""
from datetime import datetime, timezone
import gc
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/function-termination-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh GPU-hidden matched own full-size termination diagnostic required')
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tokenizers import Tokenizer
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
    from slm_lab.focused_code_continuation_data_v1 import collate_complete
    from slm_lab.function_termination_objective_v1 import backward
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.own_dense_standalone_v1 import export as standalone_export,load as standalone_load
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    tests = ROOT/'reports/public-function-contracts-and-termination-v1-final-tests.log'
    if '7 passed' not in tests.read_text():raise ValueError('Whole native/EOS/source/independent-gradient controls required')
    checkpoint = ROOT/'checkpoints/code-first-continuation-v3/step-2048/model.pt'
    source_plan = ROOT/'reports/code-first-continuation-v3/plan.json'
    source_readback = ROOT/'reports/code-first-step2048-CPU-readback-v2/complete.json'
    original_doctest_reader = ROOT/'reports/documented-doctest-verified-v1/independent-complete-source-and-runtime-readback-v1.json'
    source = json.loads(source_plan.read_text())
    if source['planned_steps'] != 7558 or not json.loads(source_readback.read_text())['complete']:
        raise ValueError('The exact fixed2048source within the original7558update run must have complete source/state readback')
    folder = ROOT/'data/documented-doctest-verified-v1'
    pack = CompleteCodePack(folder,'train','training')
    proof = json.loads(original_doctest_reader.read_text())
    if not proof['complete'] or not proof['all_reexecuted_examples_passed'] or proof['admitted_pack_sha256'] != file_hash(folder/'admission.json'):
        raise ValueError('Every original admitted function example must have independent complete execution/source readback')
    tokenizer_path = folder/'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if file_hash(tokenizer_path) != source['tokenizer_sha256']:raise ValueError('The exact own source tokenizer is required')
    registry_path = ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    registry = json.loads(registry_path.read_text())
    if len(registry) != 32953:raise ValueError('Every unchanged declared held stem must remain screened')
    overlap = HeldTextOverlap(registry)
    with gzip.open(folder/'original-rows.jsonl.gz','rt') as stream:rows = [json.loads(line) for line in stream]
    chosen = [];groups = set()
    for position in sorted(range(len(pack)),key=lambda p:(int(pack.index[pack.positions[p]]['length']),p)):
        original = rows[int(pack.positions[position])]
        ids,labels = pack.sequence(position)
        decoded = tokenizer.decode(ids[1:-1],skip_special_tokens=False)
        if original['group_id'] in groups or overlap.literal(decoded) or overlap.lexical_copies(decoded):continue
        chosen.append(position);groups.add(original['group_id'])
        if len(chosen) == 4:break
    if len(chosen) != 4:raise ValueError('Four distinct previously admitted complete source groups are required')
    selected = chosen[:2];unoptimized = chosen[2:]
    sources = {str(p):file_hash(p) for p in [checkpoint,source_plan,source_readback,original_doctest_reader,tests,registry_path,
        folder/'manifest.json',folder/'admission.json',*python_sources(ROOT,[Path(__file__)])]}
    sources.update({str(folder/name):sha for name,sha in pack.manifest['files_sha256'].items()})
    guard = FileBindings(sources)
    if shutil.disk_usage(ROOT).free < 40*1024**3+2*5*1024**3:
        raise ValueError('Both complete native diagnostic models must fit above the original writers reserve')
    OUT.mkdir();started = time.monotonic()
    settings = dict(learning_rates=dict(language=1e-5,vision=1e-4,word=3e-4),betas=[.9,.95],weight_decay=.01)
    plan = dict(at=datetime.now(timezone.utc).isoformat(),owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=sources,starting_checkpoint=str(checkpoint),starting_plan=str(source_plan),
        starting_snapshot_actual_updates=2048,starting_snapshot_is_partial_continuation_not_final_model=True,
        expected_unique_values=984399690,optimized_original_train_positions=selected,
        unoptimized_diagnostic_original_train_positions=unoptimized,original_admitted_train_population=len(pack),
        unoptimized_rows_are_not_new_held_benchmark_or_all_history_unseen=True,
        logical_source_updates_each_arm=2,arms=dict(whole_CE=0.,CE_plus_true_EOS=.05),optimizer=settings,gradient_clip=1.,
        source_example_order_and_all_native_response_EOS_targets_identical_each_arm=True,
        no_reference_model_or_external_weights_loaded=True,no_current_GPU_job_or_source_modified=True,
        production_updates=0,goal_achieved=False)
    atomic_json(OUT/'plan.json',plan);digest = file_hash(OUT/'plan.json')
    def phase(arm,text):
        atomic_json(OUT/'progress.json',dict(arm=arm,phase=text,seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
    def batch(position):return collate_complete([pack.sequence(position)],'cpu')
    def evaluate(model):
        model.eval();result = []
        with torch.no_grad():
            for position in chosen:
                b = batch(position);target = b['labels'][:,1:];mask = target != -100
                logits = model(input_ids=b['input_ids'],attention_mask=b['attention_mask'])['logits'][:,:-1].float()
                last = int(b['attention_mask'].sum())-2
                logp = logits.log_softmax(-1)
                interior = mask.clone();interior[0,last] = False
                result.append(dict(original_position=position,optimized_in_this_experiment=position in selected,
                    complete_context_tokens=int(b['attention_mask'].sum()),complete_response_EOS_targets=int(mask.sum()),
                    complete_response_EOS_CE_sum=float(F.cross_entropy(logits[mask],target[mask],reduction='sum')),
                    true_EOS_NLL=float(-logp[0,last,2]),true_EOS_probability=float(logp[0,last,2].exp()),
                    true_EOS_is_argmax=bool(logits[0,last].argmax()==2),
                    interior_response_mean_EOS_probability=float(logp[:,:,2].exp()[interior].mean()),
                    interior_response_argmax_EOS_count=int(((logits.argmax(-1)==2)&interior).sum()),
                    actual_original_input_ids_sha256=__import__('hashlib').sha256(b['input_ids'].numpy().astype('<u2').tobytes()).hexdigest()))
        return result
    arms = {};starting = None
    try:
        for arm,weight in plan['arms'].items():
            phase(arm,'loading exact own fixed2048snapshot')
            model,metadata = load(checkpoint,source_plan,'step-2048',expected_checkpoint_sha256=sources[str(checkpoint)],
                expected_plan_sha256=sources[str(source_plan)],device='cpu')
            if model.counts()['total_unique'] != 984399690 or metadata['actual_completed_updates'] != 2048:
                raise ValueError('Every actual fixed own full-size source value must load')
            initial_digest = tensor_digest(model.named_parameters())
            if starting is None:starting = initial_digest
            elif starting != initial_digest:raise ValueError('Both controlled arms must start at exactly the same actual model')
            random.seed(992317);np.random.seed(992317);torch.manual_seed(992317)
            phase(arm,'observing complete original reference and true-EOS likelihoods')
            before = evaluate(model)
            optimizer = make_optimizer(model,settings)
            groups = parameter_groups(model);language_ids = {id(p) for _,p in groups['language']}
            fixed = tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in language_ids)
            records = []
            for position in selected:
                phase(arm,'updating one complete original admitted function at position '+str(position))
                b = batch(position);model.train();model.network.gradient_checkpointing = True;optimizer.zero_grad(set_to_none=True)
                record = backward(model,b,global_targets=int((b['labels'][:,1:]!=-100).sum()),global_examples=1,EOS_weight=weight)
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _,p in groups['language']):
                    raise ValueError('Every selected actual dense language leaf must receive a finite gradient')
                record['gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_([p for _,p in groups['language']],1.,error_if_nonfinite=True))
                optimizer.step();optimizer.zero_grad(set_to_none=True);record['original_source_position'] = position;records.append(record)
            if (len(optimizer.state) != len(groups['language'])
                    or any(float(state['step']) != 2 for state in optimizer.state.values())
                    or fixed != tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in language_ids)):
                raise ValueError('All actual language Adam clocks and every inactive image/word/auxiliary value must remain exact')
            phase(arm,'observing matched post-update likelihoods without generation')
            after = evaluate(model);trained_digest = tensor_digest(model.named_parameters())
            final_Adam_digest = tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ['step','exp_avg','exp_avg_sq'])
            del optimizer,groups;gc.collect()
            target = OUT/arm;target.mkdir();shutil.copyfile(tokenizer_path,target/'tokenizer.json')
            provenance = dict(kind='two-source full-size function-EOS diagnostic',starting_snapshot_actual_updates=2048,
                partial_source_not_final_epoch=True,EOS_auxiliary_weight=weight,optimized_original_positions=selected,
                complete_source_epoch=False,generated_quality_not_measured=True)
            native_sources = dict(diagnostic_plan_sha256=digest,own_fixed2048checkpoint_sha256=sources[str(checkpoint)])
            payload = standalone_export(model,sources=native_sources,provenance=provenance,completed_updates=2,
                tokenizer_sha256=file_hash(tokenizer_path))
            atomic_checkpoint(target/'model.pt',payload,5*1024**3,40*1024**3);del payload
            probes = {p:batch(p) for p in chosen};model.eval()
            with torch.no_grad():expected = {p:model(input_ids=b['input_ids'],attention_mask=b['attention_mask'])['logits'].clone() for p,b in probes.items()}
            del model;gc.collect();phase(arm,'cold loading the complete standalone diagnostic model')
            cold,cold_meta = standalone_load(target/'model.pt',expected_checkpoint_sha256=file_hash(target/'model.pt'),expected_sources=native_sources)
            if tensor_digest(cold.named_parameters()) != trained_digest:raise ValueError('Every actual trained and fixed model value must cold-load exactly')
            with torch.no_grad():
                if any(not torch.equal(expected[p],cold(input_ids=b['input_ids'],attention_mask=b['attention_mask'])['logits']) for p,b in probes.items()):
                    raise ValueError('Every selected complete function logit tensor must cold-load bit exactly')
            arms[arm] = dict(EOS_auxiliary_weight=weight,before=before,after=after,actual_update_records=records,
                initial_model_sha256=initial_digest,trained_model_sha256=trained_digest,final_Adam_sha256=final_Adam_digest,
                all_inactive_vision_word_and_auxiliary_values_bit_exact=True,all_language_Adam_clocks_exact=True,
                actual_full_standalone_cold_values_and_four_complete_function_outputs_bit_exact=True,
                complete_native_model=str(target/'model.pt'),complete_native_model_sha256=file_hash(target/'model.pt'))
            del cold,expected,probes;gc.collect();atomic_json(OUT/(arm+'-complete.json'),arms[arm])
        if arms['whole_CE']['before'] != arms['CE_plus_true_EOS']['before']:
            raise ValueError('Every actual initial source likelihood and denominator must match between arms')
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,arms=arms,
            source_files_sha256={**sources,str(OUT/'plan.json'):digest},actual_unique_student_values=984399690,
            actual_optimized_sources_each_arm=2,actual_logical_updates_each_arm=2,unoptimized_diagnostic_sources=2,
            actual_initial_parameters_and_all_source_targets_identical=True,all_complete_standalone_cold_checks_pass=True,
            no_new_neural_generation_or_correctness_or_efficiency_gain_measured=True,
            full_source_epoch_or_GPU_capacity_or_durable_Adam_recovery_not_established=True,
            current_original_training_and_evaluation_jobs_unchanged=True,seconds=time.monotonic()-started,
            production_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_counts=984399690,
            matched_arms_complete=True,seconds=time.monotonic()-started)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            all_original_jobs_and_partial_diagnostic_evidence_preserved=True,goal_achieved=False))
        raise


if __name__ == '__main__':main()
