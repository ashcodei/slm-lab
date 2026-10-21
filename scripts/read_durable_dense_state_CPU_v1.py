"""Inspect a pinned persistent dense/Adam snapshot without initializing CUDA.

An open file descriptor pins the original inode while the trainer may replace
its checkpoint atomically. This checks saved values/source arithmetic only;
the separate actual four-GPU capacity proof supplies next-update replay.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/durable-distributed-code-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only saved-state inspection required')
    receipt=json.loads((RUN/'durable-state.json').read_text());path=Path(receipt['checkpoint'])
    import numpy as np
    import torch
    from slm_lab.code_residual_checkpoint_v1 import _validate_adam
    from slm_lab.own_code_delta_v1 import matrix_names,norm_names
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.recovered_reasoned_code_data_v1 import RecoveredReasonedCodeSFTData
    from slm_lab.serial_dense_code_epoch_v2 import schedule
    from train_durable_distributed_code_GPU_v1 import sources
    torch.set_num_threads(1);before=torch.get_rng_state().clone()
    with path.open('rb') as pinned:
        digest=hashlib.file_digest(pinned,'sha256').hexdigest()
        if digest!=receipt['sha256']:raise ValueError('Checkpoint advanced before pinning; inspect its new receipt without changing training')
        state=torch.load('/proc/self/fd/'+str(pinned.fileno()),map_location='cpu',weights_only=True,mmap=True)
        common=state['common'];step=common['completed'];model=common['model'];network=model['network']
        if (state['format']!='common-full-dense-Adam-four-private-RNG-v1' or state['world_size']!=4
                or common['format']!='own-full-dense-language-Adam-RNG-source-cursor-v1'
                or common['world_size']!=1 or common['planned']!=230 or not 1<=step<=230
                or step!=receipt['completed'] or common['cursor']!={'next_global_batch':step}
                or len(common['metrics'])!=step or model['actual_counts']['total_unique']!=984399690
                or model['config']['num_layers']!=35 or len(state['private_rngs'])!=4):
            raise ValueError('Every actual full-native source cursor and four-private-RNG inventory is required')
        study=json.loads((RUN/'study.json').read_text())
        expected=sources(study,receipt['profile'],'complete_distributed_full_language_SFT')
        if model['sources']!=expected:raise ValueError('The exact original study/profile/source bindings are required')
        names=common['optimized_names'];wanted={'embedding.weight',*matrix_names(35),*norm_names(35)}
        if set(names)!=wanted or len(names)!=317:raise ValueError('Every original dense matrix/table/norm must retain its Adam state')
        parameters=[network[name] for name in names]
        if sum(p.numel() for p in parameters)!=931244544:raise ValueError('The actual full931M language scope is required')
        _validate_adam(common['optimizer'],parameters,step)
        if network['lm_head.weight'].data_ptr()!=network['embedding.weight'].data_ptr():raise ValueError('One tied saved vocabulary is required')
        seen=set();actual=0
        for section in ['network','heads']:
            for tensor in model[section].values():
                if tensor.device.type!='cpu' or tensor.dtype!=torch.float32 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError('Every full-native saved value must remain finite CPU FP32')
                key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tensor.numel())
                if key not in seen:actual+=tensor.numel();seen.add(key)
        if actual!=984399690:raise ValueError('Every unique actual saved head/backbone value must be counted')
        rng_digests=[]
        for rng in state['private_rngs']:
            random.Random().setstate(rng['python'])
            n=rng['numpy'];np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
            torch.Generator(device='cpu').set_state(rng['torch'])
            if rng['cuda'] is None or rng['cuda'].dtype!=torch.uint8 or not rng['cuda'].numel():raise ValueError('Original private CUDA bytes must be retained')
            rng_digests.append(value_digest(rng))
        if len(set(rng_digests))!=4:raise ValueError('Four original distinct private generators are required')
        data=RecoveredReasonedCodeSFTData(ROOT);entries=schedule(data)
        if [r['actual_positions'] for r in common['metrics']]!=entries[:step]:raise ValueError('Every complete original committed source group must reconstruct')
        for i,row in enumerate(common['metrics']):
            if row['actual_Adam_step']!=i+1 or row['world_size']!=4:raise ValueError('Every common optimizer clock must reconstruct')
        pinned.seek(0)
        if hashlib.file_digest(pinned,'sha256').hexdigest()!=digest:raise ValueError('The pinned immutable checkpoint inode changed')
        if not torch.equal(before,torch.get_rng_state()) or torch.cuda.is_initialized():raise ValueError('Inspection must preserve caller RNG and avoid CUDA initialization')
        result=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),checkpoint=str(path),checkpoint_sha256=digest,
            profile=receipt['profile'],completed_updates=step,actual_native_values=actual,language_values=931244544,
            Adam_states=317,all_saved_values_and_moments_finite=True,all_original_committed_source_groups_rebuilt=True,
            private_rng_digests=rng_digests,private_CUDA_bytes_retained_not_executed=True,no_GPU_initialized=True,
            neural_training_and_CUDA_next_update_not_replayed_by_this_reader=True,goal_achieved=False)
        output=RUN/f'durable-state-{receipt["profile"]}-step-{step}-CPU-readback.json'
        with output.open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
