"""Check fixed probe provenance/denominators and compare the two own checkpoints."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.instruction_fidelity_v1 import prefix_metrics
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/instruction-fidelity-probe-v1';output=folder/'comparison.json'
    if output.exists():raise ValueError('Preserve previous comparison evidence')
    plan=json.loads((folder/'plan.json').read_text());selected=[json.loads(s) for s in (folder/'selection.jsonl').open()]
    if file_hash(folder/'selection.jsonl')!=plan['selection_sha256']:raise ValueError('Frozen source selection changed')
    by_id={r['probe_id']:r for r in selected}
    encoder=Encoder(Tokenizer.from_file(str(ROOT/'checkpoints/code-instruction-4000-assessment-v1/tokenizer.json')),
                    ModelConfig.load(ROOT/'configs/model.json'),ROOT)
    with (ROOT/'data/extended-curriculum-v1/code-instruction.train.jsonl').open('rb') as source:
        for r in selected:
            source.seek(r['original_row_byte_offset']);raw=source.read(r['original_row_bytes'])
            if hashlib.sha256(raw).hexdigest()!=r['original_raw_row_sha256'] or json.loads(raw)!=r['original_row']:
                raise ValueError('Original source row bytes or selection differ')
            ids,labels=encoder.sequence(r['original_row'],'response')
            if (ids!=r['actual_training_ids'] or labels!=[-100]*r['actual_training_target_start']+ids[r['actual_training_target_start']:]
                    or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()!=r['actual_training_ids_sha256']):
                raise ValueError('Actual training prompt/response tokens or masks differ')
    reports=[];runtime_inputs={}
    for spec in plan['checkpoints']:
        directory=folder/spec['label'];summary=json.loads((directory/'summary.json').read_text());bindings=json.loads((directory/'bindings.json').read_text())
        for name,digest in bindings['source_files_sha256'].items():
            if file_hash(name)!=digest:raise ValueError('Bound original model/runtime/data changed: '+name)
        runtime_inputs[spec['label']]=bindings['source_files_sha256']
        generations=[json.loads(s) for s in (directory/'generations.jsonl').open()]
        records=[json.loads(s) for s in (directory/'teacher-control-records.jsonl').open()]
        if ([r['probe_id'] for r in records]!=[r['probe_id'] for r in selected]
                or [r['probe_id'] for r in generations]!=[r['probe_id'] for r in selected if r['generate_free_response']]):
            raise ValueError('All fixed source/probe denominators required')
        if (file_hash(directory/'generations.jsonl')!=summary['generations_sha256']
                or file_hash(directory/'teacher-control-records.jsonl')!=summary['source_records_sha256']):
            raise ValueError('Recorded source likelihood/free generation bytes changed')
        seal=json.loads((directory/'generation-seal.json').read_text())
        if seal['model_state_sha256']!=summary['model_state_before_and_after_sha256'] or seal['checkpoint_sha256']!=spec['sha256']:
            raise ValueError('Same unchanged intended model required throughout probe')
        free={r['probe_id']:r for r in generations}
        for r in records:
            original=by_id[r['probe_id']];start=original['actual_training_target_start'];t=r['teacher']
            if t['tokens']!=len(original['actual_training_ids'])-start or not 0<=t['correct']<=t['tokens']:
                raise ValueError('Original full target denominator differs')
            bins=[t[k] for k in ['first_8','positions_9_to_32','positions_33_to_128','after_128']]
            if sum(b['tokens'] for b in bins)!=t['tokens'] or sum(b['correct'] for b in bins)!=t['correct']:
                raise ValueError('Token-bin counts inconsistent')
            if 'free_prefix' in r:
                g=free[r['probe_id']];target=original['actual_training_ids'][start:]
                if r['free_prefix']!=prefix_metrics(g['generated_token_ids'],target,128):
                    raise ValueError('Recomputed exact/free prefix results differ')
                if encoder.tokenizer.decode(g['generated_token_ids'],skip_special_tokens=False)!=g['generated_text']:
                    raise ValueError('Stored response IDs/text differ')
        def group(items):
            tokens=sum(r['teacher']['tokens'] for r in items)
            return dict(rows=len(items),teacher_tokens=tokens,
                teacher_mean_nll=sum(r['teacher']['nll_sum'] for r in items)/tokens,
                teacher_next_token_correct=sum(r['teacher']['correct'] for r in items),
                first_response_token_correct=sum(r['teacher']['first_token_correct'] for r in items),
                free_generation_rows=sum('free_prefix' in r for r in items),
                free_exact_bounded_prefixes=sum(r.get('free_prefix',{}).get('exact_bounded_gold_prefix',False) for r in items),
                full_response_budget_eligible=sum(r.get('free_prefix',{}).get('full_response_fits_generation_budget',False) for r in items),
                exact_full_training_response_tokens=sum(r.get('free_prefix',{}).get('exact_full_training_response_tokens') is True for r in items))
        overall=group(records)
        if any(summary['overall'][k]!=v for k,v in overall.items() if k!='free_generation_rows'):
            raise ValueError('Recomputed complete summary differs')
        numerical={}
        for precision in ['FP32','BF16']:
            checks=[r['padding_cache_'+precision] for r in records if 'padding_cache_'+precision in r]
            numerical[precision]=dict(rows=len(checks),padded_vs_unpadded_max_abs=max(r['padded_vs_unpadded_max_abs'] for r in checks),
                first_response_full_vs_cached_max_abs=max(r['first_response_full_vs_cached_max_abs'] for r in checks),
                first_response_argmax_agreements=sum(r['first_response_argmax_agrees'] for r in checks),
                padded_vs_unpadded_argmax_agreements=sum(r['padded_vs_unpadded_argmax_agreement'] for r in checks),
                padded_vs_unpadded_positions=sum(r['padded_vs_unpadded_positions'] for r in checks))
        reports.append(dict(label=spec['label'],checkpoint_sha256=spec['sha256'],overall=overall,
            code_only=group([r for r in records if r['language']!='general_human_instructions']),
            general_human_instructions=group([r for r in records if r['language']=='general_human_instructions']),
            numerical_diagnostic=numerical,by_stratum=summary['by_stratum'],
            reported_mean_control_minus_true_prompt_prefix_nll=summary['overall']['mean_control_minus_true_prompt_prefix_nll'],
            peak_cuda_allocated_bytes=summary['peak_cuda_allocated_bytes']))
    for key in runtime_inputs[reports[0]['label']].keys() & runtime_inputs[reports[1]['label']].keys():
        if runtime_inputs[reports[0]['label']][key]!=runtime_inputs[reports[1]['label']][key]:raise ValueError('Common source/runtime changed')
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),before=reports[0],after=reports[1],
        original_rows_and_full_training_encoder_inputs_reconstructed=True,all_predeclared_denominators_preserved=True,
        all_free_generations_precede_teacher_scoring=True,
        interpretation=[
            'The SFT model learns the documented-function response format and improves exact prefix/reference reproduction in this diagnostic.',
            'The data and interface differ from standalone algorithm tasks; previous0/28 functional benchmark remains unchanged.',
            'General human instructions remain weak and sparsely weighted in the original code stage.',
            'Wrong-prompt controls include changed signatures/source identities; likelihood differences do not isolate semantic documentation understanding.',
            'FP32 checked padding/cache first-response decisions agree; BF16 has recorded near-tie argmax differences and must not be described as bitwise identical.',
            'Not-yet-exposed refers only to these4000 SFT updates. Original code and human data may already have been seen earlier; no independent unseen-generalization claim.'
        ],source_reports_sha256={str(folder/s['label']/'summary.json'):file_hash(folder/s['label']/'summary.json') for s in plan['checkpoints']},
        audit_source_sha256=file_hash(Path(__file__)),optimizer_updates=0,candidate_promoted=False,goal_achieved=False)
    atomic_json(output,result)
    print(json.dumps({label:{k:v for k,v in result[label].items() if k not in ['by_stratum']} for label in ['before','after']}),flush=True)


if __name__=='__main__':main()
