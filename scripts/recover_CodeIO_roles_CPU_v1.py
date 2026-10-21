"""Rebuild exact original CodeI/O roles/vetoes and durably retain full ledgers.

Neither reconstructed source roles nor the protected-component screen may
change after runtime recovery. Every final ledger must reproduce its original
pre-loss SHA. Complete source execution and training admission remain separate.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from recover_public_code_registry_CPU_v1 import sha,compress,retain

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/CodeIO-role-recovery-20261021'
INDEX=Path('/tmp/slm-CodeIO-whole-source-index-v1')
SCREEN=Path('/tmp/slm-CodeIO-whole-target-screen-v2')
EXPECTED_ROLES='f88045b7e99094ecf4f6722917ee7f3d2686f21d3ca24a4e441225269aa05cca'
EXPECTED_SCREEN='8116c1acd65eb296e1c70c871ef11771a0e0c373e84e848e17215e8cb789fef6'


def run(name,label):
    with (OUT/(label+'.log')).open('x') as log:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    if result.returncode:raise ValueError('Recovered '+label+' failed; retain all partial evidence without automatic retry')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules or any(p.exists() for p in [OUT,INDEX,SCREEN]):
        raise ValueError('Fresh GPU-hidden complete-role reconstruction required')
    registry=json.loads((ROOT/'reports/public-code-registry-recovery-20261021/complete.json').read_text())
    raw=json.loads((ROOT/'reports/CodeIO-raw-recovery-20261021-v2/complete.json').read_text())
    if not registry['complete'] or not registry['original_registry_bytes_reproduced_exactly'] or not raw['complete']:
        raise ValueError('Complete unchanged original registry and pinned raw recovery are required')
    OUT.mkdir()
    # Reuse the validated lossless-copy helpers with this explicit destination.
    import recover_public_code_registry_CPU_v1 as helpers
    helpers.OUT=OUT
    sources=[Path(__file__).resolve(),ROOT/'scripts/index_CodeIO_whole_sources_CPU_v2.py',ROOT/'scripts/screen_CodeIO_whole_targets_CPU_v3.py']
    (OUT/'plan.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),owner_pid=os.getpid(),
        source_files_sha256={str(p):sha(p) for p in sources},expected_original_role_ledger_sha256=EXPECTED_ROLES,
        expected_original_complete_target_ledger_sha256=EXPECTED_SCREEN,original_complete_populations_retained=True,
        source_code_execution=False,data_admitted_to_training=False,model_updates=0),indent=2)+'\n')
    # Preserve the complete49-file upstream download ledger missed by the first collector's filename list.
    upstream=Path('/tmp/slm-public-reasoned-code-train-questions-v1/downloads-complete.json')
    retain(upstream,'original49-upstream-downloads-complete.json')
    run('index_CodeIO_whole_sources_CPU_v2.py','whole-source-index')
    role=INDEX/'every-original-source-with-fixed-role.jsonl';index=json.loads((INDEX/'complete.json').read_text())
    if not index['complete'] or index['complete_original_rows']!=227669 or sha(role)!=EXPECTED_ROLES:
        raise ValueError('Every original fixed source/component/role must reproduce exactly')
    retain(INDEX/'complete.json','index-complete.json');retain(INDEX/'plan.json','index-plan.json')
    role_copy=compress(role,'every-original-source-with-fixed-role.jsonl.gz')
    (OUT/'roles-complete.json').write_text(json.dumps(dict(complete=True,original_rows=227669,
        exact_original_role_ledger_sha256=EXPECTED_ROLES,durable_lossless_ledger=role_copy,
        full_target_screen_and_execution_and_admission_still_required=True),indent=2)+'\n')
    run('screen_CodeIO_whole_targets_CPU_v3.py','whole-code-and-cached-value-screen')
    screened=SCREEN/'every-original-source-with-complete-target-screen.jsonl'
    result=json.loads((SCREEN/'complete.json').read_text())
    if not result['complete'] or result['actual_original_rows']!=227669 or sha(screened)!=EXPECTED_SCREEN:
        raise ValueError('Every original complete-target/component veto decision must reproduce exactly')
    retain(SCREEN/'complete.json','screen-complete.json');retain(SCREEN/'plan.json','screen-plan.json')
    screen_copy=compress(screened,'every-original-source-with-complete-target-screen.jsonl.gz')
    receipt=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),complete_original_rows=227669,
        exact_original_role_ledger_sha256=EXPECTED_ROLES,exact_original_complete_target_ledger_sha256=EXPECTED_SCREEN,
        durable_original_role_ledger=role_copy,durable_original_complete_target_ledger=screen_copy,
        original_component_roles_and_every_protected_veto_unchanged=True,
        blocked_fixed_components=result['blocked_fixed_components'],counts=result['counts'],
        complete_fresh_execution_native_readback_source_review_and_admission_still_required=True,
        data_admitted_to_training=False,RL_oracle_admitted=False,new_model_updates=0,goal_achieved=False)
    with (OUT/'complete.json').open('x') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
