"""Read and verify original image bytes concurrently; mutate no source or model."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json,os,time
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/bound-image-cache-warmup-CPU-v1'
INVENTORY=ROOT/'data/codenet-replay-v1/image-files.jsonl'

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only read verification required')
    rows=list(map(json.loads,INVENTORY.read_text().splitlines()));inventory_sha=file_hash(INVENTORY)
    if len(rows)!=48032 or len({r['path'] for r in rows})!=len(rows):raise ValueError('Complete original unique image inventory required')
    OUT.mkdir();started=time.monotonic();completed=0;total=0
    atomic_json(OUT/'plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_inventory=str(INVENTORY),source_inventory_sha256=inventory_sha,files=len(rows),threads=8,
        source_files_sha256={str(Path(__file__)):file_hash(__file__),str(INVENTORY):inventory_sha},
        reads_only=True,neural_updates=0,goal_achieved=False))
    def verify(row):
        path=Path(row['path']);before=path.stat()
        if not path.is_relative_to(ROOT/'data') or file_hash(path)!=row['sha256']:raise ValueError('Original image bytes differ: '+str(path))
        after=path.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Original image changed during read')
        return after.st_size
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending=[pool.submit(verify,row) for row in rows]
            for future in as_completed(pending):
                total+=future.result();completed+=1
                if completed%500==0 or completed==len(rows):
                    atomic_json(OUT/'progress.json',dict(files_verified=completed,total_files=len(rows),actual_bytes_read=total,
                        seconds=time.monotonic()-started,neural_updates=0,goal_achieved=False))
        if file_hash(INVENTORY)!=inventory_sha:raise ValueError('Original inventory changed during read')
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),original_images_verified=completed,
            actual_bytes_read=total,seconds=time.monotonic()-started,source_or_model_mutations=0,
            cache_eviction_allowed_by_kernel=True,persistent_capacity_added=0,goal_achieved=False))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),files_verified=completed,neural_updates=0,goal_achieved=False));raise

if __name__=='__main__':main()
