"""Measure fixed train-interface likelihood, prompt dependence and free responses."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.instruction_fidelity_v1 import token_metrics, control_sequence, prefix_metrics, padding_cache_diagnostic
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(label,device):
    parent=ROOT/'reports/instruction-fidelity-probe-v1';plan_path=parent/'plan.json'
    plan=json.loads(plan_path.read_text());selection=parent/'selection.jsonl'
    exposure_path=parent/'exposure-order-independent-check.json'
    exposure=json.loads(exposure_path.read_text())
    if exposure['five_consecutive_recorded_update_target_counts_reconstructed_exactly'] is not True:
        raise ValueError('Actual saved update counts must independently confirm exposure reconstruction')
    if file_hash(selection)!=plan['selection_sha256']:raise ValueError('Fixed complete source selection required')
    specs=[s for s in plan['checkpoints'] if s['label']==label]
    if len(specs)!=1:raise ValueError('One exact predeclared own checkpoint required')
    spec=specs[0];checkpoint=Path(spec['path']);folder=parent/label
    if folder.exists():raise ValueError('Preserve previous outputs/partial diagnostics')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']
    if shutil.disk_usage(ROOT).free<reserve+64*1024**2:raise ValueError('Remaining writer reserve plus64MiB probe budget required')
    dev=torch.device(device);torch.cuda.set_device(dev)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(dev)[0]<12*1024**3:
        raise ValueError('Initially12GiB free BF16-capable shared CUDA capacity required')
    torch.cuda.set_per_process_memory_fraction(.14,dev);torch.set_num_threads(2)
    paths=[Path(__file__),ROOT/'src/slm_lab/instruction_fidelity_v1.py',
        ROOT/'scripts/prepare_codenet_native_contract_pack_v2.py',ROOT/'src/slm_lab/codenet_native_contracts_v1.py',
        ROOT/'src/slm_lab/recurrent_ablation.py',ROOT/'src/slm_lab/corpus.py',ROOT/'src/slm_lab/diverse_images.py',
        plan_path,selection,exposure_path,checkpoint,checkpoint.parent/'tokenizer.json']
    bound={**plan['source_files_sha256'],**{str(p):file_hash(p) for p in paths}}
    if bound[str(checkpoint)]!=spec['sha256']:raise ValueError('Exact predeclared own model bytes required')
    guard=FileBindings(bound);rows=[json.loads(s) for s in selection.open()]
    by_id={r['probe_id']:r for r in rows}
    if len(rows)!=128 or len(by_id)!=128 or sum(r['generate_free_response'] for r in rows)!=64:
        raise ValueError('Every predeclared diagnostic row required')
    folder.mkdir()
    def status(message,**fields):
        atomic_json(folder/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            optimizer_updates=0,candidate_promoted=False,goal_achieved=False,**fields))
    def storage_guard():
        if sum(p.stat().st_size for p in parent.rglob('*') if p.is_file())>64*1024**2 or shutil.disk_usage(ROOT).free<512*1024**2:
            raise ValueError('Probe report budget exceeded or filesystem almost full')
    try:
        status('loading fixed own model for training-interface diagnostic',checkpoint=label)
        model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
        if sum(p.numel() for p in model.parameters())!=706012672:
            raise ValueError('Actual own706M model required')
        tok=Tokenizer.from_file(str(checkpoint.parent/'tokenizer.json'))
        if meta['tokenizer_sha256']!=file_hash(checkpoint.parent/'tokenizer.json'):
            raise ValueError('Original own tokenizer required')
        before=state_digest(model.state_dict());torch.cuda.reset_peak_memory_stats(dev)
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
        atomic_json(folder/'bindings.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
            checkpoint_sha256=spec['sha256'],source_files_sha256=bound,checkpoint_metadata={k:meta.get(k) for k in ['stage','steps','planned_steps','world_size','data_sha256']},
            cuda_device=device,memory_fraction=.14,occupied_verified_writer=occupied,
            free_generation_before_any_teacher_scoring=True,goal_achieved=False))
        generated={};gen_path=folder/'generations.jsonl'
        with gen_path.open('x') as stream:
            for row in rows:
                if not row['generate_free_response']:continue
                guard.check();start=row['actual_training_target_start'];prefix=row['actual_training_ids'][:start]
                record=dict(probe_id=row['probe_id'],generation_budget=128,prompt_tokens=len(prefix))
                if len(prefix)+128>model.config.max_text_length:
                    record.update(assessed=False,reason='Whole actual training prompt plus128-token generation budget does not fit')
                else:
                    ids=torch.tensor([prefix],device=dev)
                    with amp():out=model.generate(ids,max_new_tokens=128,eos_token_id=tok.token_to_id('<eos>'),use_cache=True)
                    response=out[0,len(prefix):].tolist()
                    record.update(assessed=True,generated_token_ids=response,generated_text=tok.decode(response,skip_special_tokens=False),
                        stop_reason='eos' if response[-1]==tok.token_to_id('<eos>') else 'token_budget')
                generated[row['probe_id']]=record;stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                if len(generated)%8==0:
                    storage_guard();status('generating fixed source-interface responses before teacher scoring',generated=len(generated),required=64)
                    print('free_generations',len(generated),flush=True)
            os.fsync(stream.fileno())
        if len(generated)!=64 or state_digest(model.state_dict())!=before:raise ValueError('Complete free generations with unchanged own weights required')
        atomic_json(folder/'generation-seal.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
            generations_sha256=file_hash(gen_path),checkpoint_sha256=spec['sha256'],model_state_sha256=before,
            prompts_only_no_gold_responses_given_to_free_generation=True,optimizer_updates=0,goal_achieved=False))
        records=[]
        with (folder/'teacher-control-records.jsonl').open('x') as stream:
            for row in rows:
                guard.check();ids=torch.tensor([row['actual_training_ids']],device=dev);start=row['actual_training_target_start']
                with amp():result=token_metrics(model(ids)['logits'],ids,start)
                record=dict(probe_id=row['probe_id'],language=row['language'],exposure_phase=row['exposure_phase'],teacher=result)
                try:control,start_control=control_sequence(row,by_id[row['shuffled_prompt_control_id']],model.config.max_text_length)
                except ValueError as error:record['control_unassessable']=str(error)
                else:
                    control_ids=torch.tensor([control],device=dev)
                    with amp():record['shuffled_prompt_control']=token_metrics(model(control_ids)['logits'],control_ids,start_control)
                    bins=[result[k] for k in ['first_8','positions_9_to_32','positions_33_to_128']]
                    same_tokens=sum(b['tokens'] for b in bins)
                    if record['shuffled_prompt_control']['tokens']!=same_tokens:raise ValueError('Matched control response targets required')
                    record['control_minus_true_prompt_prefix_mean_nll']=(record['shuffled_prompt_control']['nll_sum']-sum(b['nll_sum'] for b in bins))/same_tokens
                if row['generate_free_response']:
                    free=generated[row['probe_id']]
                    if free['assessed']:
                        target=row['actual_training_ids'][start:]
                        record['free_prefix']=prefix_metrics(free['generated_token_ids'],target,128)
                        record['teacher_vs_cached_first_response_token_agrees']=result['first_token_prediction_id']==free['generated_token_ids'][0]
                    else:record['free_generation_unassessable']=free['reason']
                if row['audit_padding_and_cache']:
                    if ids.shape[1]+7>model.config.max_text_length:record['padding_unassessable']='Full source context plus7 pad tokens exceeds the model window'
                    else:
                        record['padding_cache_FP32']=padding_cache_diagnostic(model,ids,start,nullcontext)
                        record['padding_cache_BF16']=padding_cache_diagnostic(model,ids,start,amp)
                records.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
                if len(records)%16==0:
                    storage_guard();status('measuring fixed full teacher responses, shuffled-prompt controls and padding/cache paths',scored=len(records),required=128)
                    print('teacher_control_rows',len(records),flush=True)
        guard.check(full=True)
        if state_digest(model.state_dict())!=before:raise ValueError('Diagnostic changed own model state')
        def summarize(items):
            n=len(items);teacher=sum(r['teacher']['tokens'] for r in items)
            free=[r['free_prefix'] for r in items if 'free_prefix' in r]
            control=[r['control_minus_true_prompt_prefix_mean_nll'] for r in items if 'control_minus_true_prompt_prefix_mean_nll' in r]
            return dict(rows=n,teacher_tokens=teacher,teacher_mean_nll=sum(r['teacher']['nll_sum'] for r in items)/teacher,
                teacher_next_token_correct=sum(r['teacher']['correct'] for r in items),
                first_response_token_correct=sum(r['teacher']['first_token_correct'] for r in items),
                controls_assessed=len(control),mean_control_minus_true_prompt_prefix_nll=sum(control)/len(control) if control else None,
                free_generations_assessed=len(free),free_gold_prefix_tokens=sum(r['compared_gold_prefix_tokens'] for r in free),
                free_position_matched_tokens=sum(r['position_matched_tokens'] for r in free),
                free_exact_bounded_prefixes=sum(r['exact_bounded_gold_prefix'] for r in free),
                full_response_budget_eligible=sum(r['full_response_fits_generation_budget'] for r in free),
                exact_full_training_response_tokens=sum(r['exact_full_training_response_tokens'] is True for r in free),
                teacher_vs_cached_first_response_token_agreements=sum(r.get('teacher_vs_cached_first_response_token_agrees',False) for r in items))
        summary=dict(completed_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=spec['sha256'],
            overall=summarize(records),by_stratum={language+':'+phase:summarize([r for r in records if r['language']==language and r['exposure_phase']==phase])
                for language in sorted({r['language'] for r in records}) for phase in ['exposed','not_yet_exposed']},
            padding_cache_records=[{k:v for k,v in r.items() if k.startswith('padding_') or k=='probe_id'} for r in records if 'padding_cache_FP32' in r or 'padding_unassessable' in r],
            fixed_tolerances=dict(FP32_max_abs=.002,BF16_max_abs=.5),
            source_records_sha256=file_hash(folder/'teacher-control-records.jsonl'),generations_sha256=file_hash(gen_path),
            model_state_before_and_after_sha256=before,all_model_state_bytes_unchanged=True,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(dev),optimizer_updates=0,candidate_promoted=False,goal_achieved=False,
            scope=plan['interpretation']+' Teacher-forced prediction and128-token prefix matching are not functional correctness or independent generalization. All free generations precede use of gold responses for scoring.')
        atomic_json(folder/'summary.json',summary);status('complete instruction-fidelity diagnostic; no capability promotion',summary=summary['overall'])
        print(json.dumps(summary['overall']),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    except Exception as error:
        status('instruction-fidelity diagnostic stopped; preserve partial outputs',error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',required=True);parser.add_argument('--device',required=True)
    args=parser.parse_args();run(args.label,args.device)


if __name__=='__main__':main()
