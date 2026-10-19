"""Four actual CPU/Gloo ranks test the production partial-batch reducer."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/fast-joint-four-rank-CPU-reduction-v1.json'


def worker(rank,world,folder):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU collective check must hide every GPU')
    import torch
    from torch import distributed as dist
    from slm_lab.fast_joint_stage_v1 import reduce_gradients
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=world)
    try:
        a=torch.nn.Parameter(torch.zeros(19));b=torch.nn.Parameter(torch.zeros(5));unused=torch.nn.Parameter(torch.zeros(7))
        if rank<3:a.grad=torch.full_like(a,(rank+1)/6.)
        if rank in (0,2):b.grad=torch.full_like(b,rank+1.)
        active=reduce_gradients([a,b,unused],'cpu',world,bucket_values=8)
        if active!=2 or not torch.allclose(a.grad,torch.ones_like(a),atol=1e-7,rtol=0) or not torch.equal(b.grad,torch.full_like(b,4.)) or unused.grad is not None:
            raise ValueError('Partial-rank globally normalized sum or inactive-head behavior differs')
        (Path(folder)/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,active=active,partial_batch_gradients_exact=True,unused_head_grad_is_none=True)))
    finally:dist.destroy_process_group()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Every GPU must be hidden')
    if REPORT.exists():raise ValueError('Preserve previous real collective evidence')
    import torch.multiprocessing as mp
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='slm-joint-reduction-') as tmp:
        mp.spawn(worker,args=(4,tmp),nprocs=4,join=True)
        rows=[json.loads((Path(tmp)/f'rank-{r}.json').read_text()) for r in range(4)]
    atomic_json(REPORT,dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_ranks=4,backend='gloo',all_ranks_passed=True,
        nonempty_and_empty_final_partial_ranks_checked=True,global_normalization_not_divided_by_world_again=True,
        inactive_word_or_vision_leaves_retain_grad_none=True,proofs=rows,seconds=time.monotonic()-started,
        source_files_sha256={str(Path(__file__).resolve()):file_hash(__file__),str(ROOT/'src/slm_lab/fast_joint_stage_v1.py'):file_hash(ROOT/'src/slm_lab/fast_joint_stage_v1.py')},
        actual_GPU_NCCL_capacity_and_learning_are_separate=True,goal_achieved=False))


if __name__=='__main__':main()
