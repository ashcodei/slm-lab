"""Reconcile actual recent token exposure and measured local training pace.

This deliberately does not invent an all-ancestry token count, count repeated
presentations as unique data, or treat a scaling guideline as a capability
guarantee. It constructs no model/optimizer and changes no training process.
"""
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/recent-pretraining-scale-CPU-v1'


def read(path):return json.loads(Path(path).read_text())


def totals(rows):
    result=Counter();by_family={}
    for i,row in enumerate(rows):
        if row['step']!=i+1:raise ValueError('Every exact original source journal step is required')
        item=by_family.setdefault(row['family'],Counter())
        values=dict(updates=1,complete_context_tokens=row['complete_context_tokens'],response_EOS_targets=int(row['response_EOS_targets']),
            language_optimized_response_EOS_targets=int(row['response_EOS_targets']) if 'language' in row['actual_active_groups'] else 0)
        item.update(values);result.update(values)
    return dict(total=dict(result),by_family={k:dict(v) for k,v in by_family.items()})


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden recent-token audit required')
    import torch
    torch.set_num_threads(1)
    main_dir=ROOT/'reports/phase-joint-v3';main_plan=read(main_dir/'plan.json')
    reader_path=ROOT/'reports/completed-phase-joint-v3-CPU-readback-v1/complete.json';reader=read(reader_path)
    if not reader['complete'] or reader['actual_original_updates']!=35284:raise ValueError('Exact independently reconciled complete main epoch required')
    journal=main_dir/'training-metrics.jsonl'
    if file_hash(journal)!=reader['source_files_sha256'][str(journal)]:raise ValueError('The original complete main journal must remain unchanged')
    main_rows=[json.loads(line) for line in journal.open()];main=totals(main_rows)
    if main['total']['updates']!=35284:raise ValueError('All35284actual main updates are required')
    for name,value in reader['actual_source_budget'].items():
        if any(main['by_family'][name][k]!=value[k] for k in ['complete_context_tokens','response_EOS_targets']):
            raise ValueError('Every reconciled actual family context/target budget must match')
    current_dir=ROOT/'reports/code-first-continuation-v3';plan_path=current_dir/'plan.json';plan=read(plan_path)
    progress=read(current_dir/'progress.json');completed=progress['actual_updates'];raw=[]
    with (current_dir/'training-metrics.jsonl').open('rb') as stream:
        for _ in range(completed):
            line=stream.readline()
            if not line.endswith(b'\n'):raise ValueError('Every actually reported committed source row must be fully flushed')
            raw.append(line)
    current_rows=[json.loads(line) for line in raw]
    if current_rows[-1]!=progress['latest']:raise ValueError('The exact atomic progress snapshot and committed journal prefix must agree')
    current=totals(current_rows);selected=plan['selected_source_budget'];planned={}
    for key in ['complete_context_tokens','response_EOS_targets']:planned[key]=sum(v[key] for v in selected.values())
    planned['updates']=sum(v['updates'] for v in selected.values())
    if planned['updates']!=7558:raise ValueError('The exact full declared continuation budget must remain intact')
    checkpoint=ROOT/'checkpoints/phase-joint-v3/initial/model.pt'
    if file_hash(checkpoint)!=reader['source_files_sha256'][str(checkpoint)]:raise ValueError('The exact complete counted own architecture is required')
    native=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);payload=native['model']
    counts=dict(language=0,vision=0,word=0,other=0)
    for group in ['network','heads']:
        for name,value in payload[group].items():
            if group=='network' and name=='lm_head.weight':continue
            kind=('vision' if group=='network' and name.startswith('vision.') else
                  'language' if group=='network' and not name.startswith('value_head.') else
                  'word' if group=='heads' and name.startswith('word.') else 'other')
            counts[kind]+=value.numel()
    student=sum(counts.values())
    if student!=984399690:raise ValueError('Every actual unique student value must fit the fixed own sub1B architecture')
    seconds=progress['elapsed_seconds'];rate=current['total']['complete_context_tokens']/seconds
    observed_targets=main['total']['response_EOS_targets']+current['total']['response_EOS_targets']
    OUT.mkdir();atomic_json(OUT/'captured-current-progress.json',progress)
    (OUT/'captured-current-journal-prefix.jsonl').write_bytes(b''.join(raw))
    sources=[Path(__file__),main_dir/'plan.json',reader_path,journal,checkpoint,plan_path,
             OUT/'captured-current-progress.json',OUT/'captured-current-journal-prefix.jsonl']
    result=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,scope='Only the complete35284update main and the actually observed current continuation prefix',
        actual_unique_student_values=student,actual_parameter_values_by_partition=counts,
        completed_main_actual_exposure=main,current_observed_actual_exposure=current,full_current_declared_exposure=planned,
        actual_current_updates=completed,planned_current_updates=7558,
        recent_two_run_supervised_targets_so_far=observed_targets,
        recent_two_run_targets_per_counted_student_value=observed_targets/student,
        complete_main_plus_full_current_planned_targets=main['total']['response_EOS_targets']+planned['response_EOS_targets'],
        all_model_ancestry_token_exposure_not_reconstructed=True,repeated_presentations_not_unique_document_tokens=True,
        vision_or_word_only_updates_not_counted_as_language_optimized_targets=True,
        observed_current_wall_seconds=seconds,observed_current_context_tokens_per_wall_second=rate,
        illustrative_additional_context_token_pace_projections={str(n):dict(seconds=n/rate,days=n/rate/86400) for n in [10**9,10**10,2*10**10]},
        pace_projection_not_a_guarantee_or_new_GPU_reservation=True,
        current_mix_checkpoint_validation_reference_and_IO_costs_included=True,
        undertraining_is_a_supported_hypothesis_not_a_completed_causal_ablation=True,
        primary_research_context=[dict(url='https://arxiv.org/abs/2203.15556',finding='Compute-optimal training depends jointly on parameter count and training-token count.'),
            dict(url='https://arxiv.org/abs/2502.02737',finding='The authors report about11trillion pretraining tokens for their1.7Bsmall language model, using staged web/math/code mixtures.')],
        research_models_not_used_as_weight_initializers_or_live_teachers=True,
        small_current_budget_or_more_data_does_not_prove_or_guarantee_frontier_equivalence=True,
        source_files_sha256={str(p):file_hash(p) for p in sources},models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',result)
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_current_updates=completed,
        main_actual_targets=main['total']['response_EOS_targets'],full_current_planned_targets=planned['response_EOS_targets'],
        observed_context_tokens_per_wall_second=rate,all_ancestry_total_not_established=True)),flush=True)


if __name__=='__main__':main()
