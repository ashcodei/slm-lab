"""Explicit measured coding-priority handover with durable prior state retained."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal

ROOT=Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU handover must hide all GPUs')
    import assess_joint_java_comparison_v1 as resources
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    destination=ROOT/'reports/verified-views-execution-v1-handover.json'
    if destination.exists():raise ValueError('Preserve earlier explicit handover')
    previous=ROOT/'reports/protected-joint-v1';plan=json.loads((previous/'plan.json').read_text())
    queue=json.loads((ROOT/'reports/verified-views-execution-v1-queue.json').read_text())
    if not resources.same(queue['owner']) or queue['predecessor']!=plan['owner']:raise ValueError('Exact prepared successor must be live and waiting for this exact predecessor')
    FileBindings(queue['source_files_sha256']).check(full=True)
    readback=ROOT/'reports/protected-joint-step2048-readback-v1/complete.json'
    scores=ROOT/'reports/protected-joint-development2048-readback-v1.json'
    state=json.loads(readback.read_text());quality=json.loads(scores.read_text())
    if (state['actual_completed_updates']!=2048 or state['actual_Adam_states']!=445
            or not state['actual_native_weights_equal_resumable_weights'] or not quality['persistent_coding_degradation']):
        raise ValueError('Actual preserved2048-state/Adam and persistent matched code degradation required')
    matched=quality['matched_development_scores']
    if any(matched[a]['correct']!=0 or matched[a]['syntax']>matched['initial']['syntax']-5 for a in ['step-1024','step-2048']):
        raise ValueError('Both measured checkpoints must have zero coding correctness and large syntax losses')
    bindings={str(readback):file_hash(readback),str(scores):file_hash(scores)}
    snapshot=ROOT/'reports/protected-joint-step2048-readback-v1/resume-step2048.pt'
    for path in [snapshot,*[ROOT/'checkpoints/protected-joint-v1'/a/'model.pt' for a in ['initial','step-1024','step-2048']],
            ROOT/'reports/verified-views-warmup-full-CPU-v1/complete.json',ROOT/'reports/verified-views-execution-v1-tests.log',
            ROOT/'reports/causal-code-rewards-v2/admission.json',Path(__file__)]:bindings[str(path)]=file_hash(path)
    raw=(previous/'training-metrics.jsonl').read_bytes();complete=raw[:raw.rfind(b'\n')+1];records=list(map(json.loads,complete.splitlines()))
    actual=resources.same(plan['owner']);argv=(Path('/proc')/str(plan['owner']['pid'])/'cmdline').read_bytes().split(b'\0')
    if not actual or not any(Path(a.decode()).name=='train_protected_joint_v1.py' for a in argv if a):raise ValueError('Exact old owner birth and original controller command required immediately before handover')
    record=dict(at=datetime.now(timezone.utc).isoformat(),retired_exact_owner=plan['owner'],prepared_successor_exact_owner=queue['owner'],
        retained_completed_updates=2048,retained_complete_Adam_states=445,retained_resumable_state=str(snapshot),
        retained_evidence_sha256=bindings,initial_and_all_evaluated_checkpoints_preserved=True,
        partial_epoch_and_endpoints_explicitly_incomplete=True,original_broad_training_and_genuine_RLHF_scope_still_required=True,
        matched_development_scores=matched,observed_completed_update_records_before_stop=len(records),
        observed_last_update_before_stop=records[-1]['step'],observed_complete_journal_prefix_sha256=hashlib.sha256(complete).hexdigest(),
        later_uncheckpointed_weights_not_claimed_retained=True,
        decision='Manual coding-priority research handover after two zero-correctness, strongly syntax-regressing checkpoints. '
            'One extra knowledge answer and recovery of the small math score are retained findings; no all-front automatic-stop claim is made. '
            'The successor changes training signal to exact-source completion/repair warm-up and controlled whole-stream execution RL. '
            'It must prove its own full configured four-L40 capacity before any production updates.',
        predecessor_not_promoted=True,automatic_retry=False,goal_achieved=False)
    atomic_json(destination,record)
    if not resources.same(plan['owner']):raise ValueError('Old owner ended between review and signal; receipt preserved for explicit review')
    os.kill(plan['owner']['pid'],signal.SIGTERM)
    print(json.dumps(dict(retired_exact_owner=plan['owner'],retained_updates=2048,actual_observed_updates=len(records),successor=queue['owner'])),flush=True)


if __name__=='__main__':main()
