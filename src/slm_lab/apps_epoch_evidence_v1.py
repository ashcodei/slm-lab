"""Independent original-program/rehearsal exposure accounting for a saved epoch."""
from collections import Counter
import math
import random


def exposures(metrics,code_records,populations,*,seed,planned_steps,world=4,code_accumulation=8,
        replay_accumulation=2,replay_coefficient=.5,sampler_state=None,complete=False,replay_target_count=None):
    if (not metrics or not code_records or len(dict(populations))!=len(populations)
            or any(type(size) is not int or size<1 for _,size in populations)
            or len(metrics)>planned_steps or (complete and len(metrics)!=planned_steps)):
        raise ValueError('Exact nonempty original source populations and declared update span required')
    width=world*code_accumulation;replay_width=world*replay_accumulation
    if planned_steps!=(len(code_records)+width-1)//width:
        raise ValueError('Declared budget must be one complete original program pass with rounding only')
    permutations={}
    def at(size,initial,position):
        epoch,offset=divmod(position,size);key=(size,initial,epoch)
        if key not in permutations:
            values=list(range(size));random.Random(initial+epoch).shuffle(values);permutations[key]=values
        return permutations[key][offset]
    names=[name for name,_ in populations];sizes=dict(populations);counts=Counter();used={n:set() for n in names}
    code_seen=Counter();code_targets=0;replay_targets=Counter()
    for step,metric in enumerate(metrics):
        name=names[step%len(names)];occurrence=step//len(names)
        field=None if name=='original_packed_code' else 'response'
        if (metric['step']!=step+1 or metric['replay_family']!=name or metric['replay_occurrence']!=occurrence
                or metric['replay_response_field']!=field or len(metric['worker_source_positions'])!=world):
            raise ValueError('Every exact ordered original optimizer step/family/occurrence/field required')
        actual=metric['worker_source_positions']
        if any(len(p['code_positions'])!=code_accumulation or len(p['replay_positions'])!=replay_accumulation for p in actual):
            raise ValueError('All actual worker/microbatch source positions required')
        codes=[actual[rank]['code_positions'][micro] for micro in range(code_accumulation) for rank in range(world)]
        previous=[actual[rank]['replay_positions'][micro] for micro in range(replay_accumulation) for rank in range(world)]
        expected_code=[at(len(code_records),seed,step*width+i) for i in range(width)]
        expected_replay=[at(sizes[name],seed+1000+step%len(names),occurrence*replay_width+i) for i in range(replay_width)]
        if codes!=expected_code or previous!=expected_replay:
            raise ValueError('Actual original code and rehearsal epoch permutations/draws differ')
        code_seen.update(codes);used[name].update(previous);counts[name]+=1
        objectives=metric['objectives']
        if set(objectives)!={'complete_original_human_program',name}:
            raise ValueError('Exactly the declared original-program and rehearsal objectives required')
        n=sum(code_records[i]['target_tokens'] for i in codes)
        if objectives['complete_original_human_program']['global_target_tokens']!=n:
            raise ValueError('Every original human response/EOS target exposure must reconstruct')
        code_targets+=n
        n_replay=objectives[name]['global_target_tokens']
        if replay_target_count is not None and n_replay!=sum(replay_target_count(name,i,field) for i in previous):
            raise ValueError('Every actual source-encoded rehearsal target exposure must reconstruct')
        replay_targets[name]+=n_replay
        for objective,coefficient in [('complete_original_human_program',1.),(name,replay_coefficient)]:
            value=objectives[objective]
            if (value['coefficient']!=coefficient or type(value['global_target_tokens']) is not int or value['global_target_tokens']<1
                    or not math.isfinite(value['token_mean_nll']) or value['token_mean_nll']<0
                    or not math.isfinite(value['weighted_objective'])
                    or not math.isclose(value['weighted_objective'],coefficient*value['token_mean_nll'],rel_tol=1e-12,abs_tol=1e-12)):
                raise ValueError('Finite correctly weighted original token-mean objectives required')
        if (metric['total_target_tokens']!=n+n_replay or not math.isclose(metric['summed_objective'],
                sum(v['weighted_objective'] for v in objectives.values()),rel_tol=1e-12,abs_tol=1e-12)):
            raise ValueError('Actual whole update target/objective totals differ')
    draws=len(metrics)*width
    expected_state=dict(updates=len(metrics),code_global_draws=draws,code_epoch_and_cursor=list(divmod(draws,len(code_records))),
        source_population=len(code_records),replay={n:dict(updates=counts[n],global_draws=counts[n]*replay_width,population=sizes[n]) for n in names},
        full_prior_source_pass=False)
    if sampler_state is not None and sampler_state!=expected_state:
        raise ValueError('Every original code and per-family rehearsal cursor/population must match observed draws')
    if complete and set(code_seen)!=set(range(len(code_records))):
        raise ValueError('Every admitted original program must be exposed in the completed epoch')
    return dict(observed_updates=len(metrics),planned_updates=planned_steps,original_program_exposures=draws,
        distinct_original_program_positions=len(code_seen),rounding_or_prefix_repeats=sum(n-1 for n in code_seen.values()),
        actual_original_program_response_EOS_targets=code_targets,full_original_program_pass=complete,
        per_rehearsal_family={n:dict(population=sizes[n],updates=counts[n],logical_record_exposures=counts[n]*replay_width,
            distinct_logical_positions=len(used[n]),target_exposures=replay_targets[n]) for n in names},
        every_original_code_and_rehearsal_position_independently_reconstructed=True,
        every_code_response_EOS_target_independently_reconstructed=True,
        every_rehearsal_target_source_reencoded=replay_target_count is not None,
        every_original_source_cursor_and_objective_weight_reconciled=sampler_state is not None,
        full_prior_source_pass=False,model_capability_gain_established=False,goal_achieved=False)
