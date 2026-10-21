"""Reacquire pinned raw CodeI/O and keep its critical receipt on persistent disk.

The original unadmitted download/inventory implementation is unchanged.
Raw source stays on scratch; publisher attribution, exact source hashes and
complete inventory survive a later loss of that scratch filesystem.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/tmp/slm-CodeIO-raw-candidate-v1')
OUT=ROOT/'reports/CodeIO-raw-recovery-20261021'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules or OUT.exists() or RAW.exists():
        raise ValueError('Fresh GPU-hidden data-only recovery and output paths required')
    OUT.mkdir()
    driver=ROOT/'scripts/fetch_CodeIO_raw_candidate_v1.py'
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner_pid=os.getpid(),
        acquisition_source=str(driver),acquisition_source_sha256=sha(driver),
        raw_source_on_volatile_scratch=True,critical_original_publisher_receipts_durable=True,
        source_code_execution=False,training_admission=False,model_updates=0)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (OUT/'acquisition.log').open('x') as log:
        result=subprocess.run([sys.executable,'-u',str(driver)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','TOKENIZERS_PARALLELISM':'false'})
    if result.returncode:
        for name in ['plan.json','progress.json','failure.json']:
            if (RAW/name).exists():shutil.copyfile(RAW/name,OUT/('acquisition-'+name))
        raise ValueError('Original source acquisition failed; all partial evidence is retained')
    receipt=json.loads((RAW/'complete.json').read_text());source=Path(receipt['original_source_file'])
    if (not receipt['complete'] or receipt['original_rows']!=227669
            or receipt['original_source_bytes']!=988821246
            or receipt['original_source_sha256']!='e6f0b1d1db181764a5d6ca6d2dcf8b641b63c632468672d1abfc24837700cca4'
            or sha(source)!=receipt['original_source_sha256'] or sha(driver)!=plan['acquisition_source_sha256']):
        raise ValueError('Every unchanged pinned source byte, inventory and acquisition source is required')
    retained={}
    for name in ['plan.json','complete.json','publisher-metadata.json','publisher-README.md']:
        target=OUT/('acquisition-'+name)
        shutil.copyfile(RAW/name,target)
        if sha(target)!=sha(RAW/name):raise ValueError('Every retained receipt and publisher attribution byte must match')
        retained[str(target)]=sha(target)
    complete=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),
        original_source_file=str(source),original_source_sha256=receipt['original_source_sha256'],
        original_source_bytes=source.stat().st_size,original_rows=receipt['original_rows'],
        rows_with_cached_IO=receipt['rows_with_cached_IO'],cached_IO_pairs=receipt['cached_IO_pairs'],
        retained_original_receipts_sha256=retained,raw_source_still_volatile=True,
        roles_full_held_screen_execution_source_review_and_native_admission_still_required=True,
        no_original_source_code_or_loader_executed=True,data_admitted_to_training=False,
        RL_reward_oracle_admitted=False,actual_model_updates=0,goal_achieved=False)
    target=OUT/'complete.json'
    with target.open('x') as stream:
        json.dump(complete,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({k:v for k,v in complete.items() if k!='retained_original_receipts_sha256'}),flush=True)


if __name__=='__main__':main()
