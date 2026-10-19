"""Read-only endpoint weights/exposures and matched native capability report.

The existing checkpoint reader is reused with a declared output redirection;
its historical readback artifact is never overwritten. No GPU contexts, model
updates, stopping decisions, resource signals or policy promotions occur here.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import torch

import train_apps_full_language_v1 as owner
from slm_lab.apps_epoch_evidence_v1 import exposures
from slm_lab.apps_native_metrics_v1 import NAMES,matched,summarize
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=owner.ROOT
FOLDER=ROOT/'reports/apps-full-language-endpoint-v1'
QUEUE=ROOT/'reports/apps-full-language-endpoint-v1-queue.json'


def source_exposures(plan,metrics,sampler=None,complete=False):
    source,replay,_,encoder=owner.open_inputs(plan)
    populations=[(name,replay.size(name)) for name in replay.families]
    checks=json.loads(owner.SOURCE_CHECK.read_text())
    if (list(replay.families)!=checks['actual_source_families']
            or populations!=[(r['family'],r['size']) for r in checks['selected_actual_replay_batches']]):
        raise ValueError('Every actual original rehearsal population must match the source audit')
    cache={}
    def targets(name,position,field):
        key=(name,position)
        if key not in cache:
            ids,labels=encoder.sequence(replay.row(name,position),field)
            if len(ids)>4096 or len(ids)!=len(labels):raise ValueError('Every whole original rehearsal context required')
            cache[key]=sum(t!=-100 for t in labels[1:])
        return cache[key]
    options=plan['training_options']
    result=exposures(metrics,source.records,populations,seed=options['seed'],planned_steps=3203,
        code_accumulation=options['code_accumulation'],replay_accumulation=options['replay_accumulation'],
        replay_coefficient=options['replay_coefficient'],sampler_state=sampler,complete=complete,replay_target_count=targets)
    if encoder.truncated:raise ValueError('No actual source sequence may be clipped during independent reconstruction')
    source.guard.check(full=True)
    result['source_reencoded_distinct_rehearsal_positions']=len(cache)
    return result


def prefix():
    target=ROOT/'reports/apps-full-language-prefix-exposure-v1.json'
    snapshot=target.with_suffix('.jsonl.gz')
    if target.exists() or snapshot.exists():raise ValueError('Preserve original prefix exposure evidence')
    plan=json.loads((owner.REPORTS/'plan.json').read_text())
    if not owner.resources.same(plan['owner']):raise ValueError('Exact original optimizer owner must be live for this prefix observation')
    data=(owner.REPORTS/'training-metrics.jsonl').read_bytes()
    if not data.endswith(b'\n'):data=data.rsplit(b'\n',1)[0]+b'\n'
    metrics=list(map(json.loads,data.splitlines()))
    snapshot.write_bytes(gzip.compress(data,mtime=0))
    result=source_exposures(plan,metrics)
    result.update(observed_at=datetime.now(timezone.utc).isoformat(),metric_snapshot_sha256=file_hash(snapshot),
        metric_snapshot_uncompressed_sha256=hashlib.sha256(data).hexdigest(),run_plan_sha256=file_hash(owner.REPORTS/'plan.json'),
        observed_original_owner=owner.resources.same(plan['owner']),weights_read_back_at_this_prefix=False,
        implementation_sha256=file_hash(__file__),independent_exposure_implementation_sha256=file_hash(ROOT/'src/slm_lab/apps_epoch_evidence_v1.py'))
    atomic_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_rehearsal_family'},indent=2),flush=True)


def checkpoint(plan):
    target=FOLDER/'complete-neural-and-source-readback.json'
    if target.exists():raise ValueError('Preserve completed endpoint checkpoint readback')
    script=ROOT/'scripts/readback_apps_full_language_v1.py'
    spec=importlib.util.spec_from_file_location('owned_original_APPS_reader',script)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    captured=[]
    def capture_only(path,value):
        if Path(path)!=ROOT/'reports/apps-full-language-v1-neural-readback.json' or captured:
            raise ValueError('Only the original single readback output may be redirected')
        captured.append(value)
    reader.atomic_json=capture_only
    with redirect_stdout(io.StringIO()):reader.main()
    if len(captured)!=1 or captured[0]['actual_saved_steps']!=3203:
        raise ValueError('Only the actual final3203update checkpoint can supply endpoint proof')
    result=captured[0]
    resume=owner.OUTPUT/'resume.pt';export=owner.OUTPUT/'model.pt'
    if file_hash(resume)!=result['checkpoint_sha256']:raise ValueError('Actual final resume file changed after original readback')
    state=torch.load(resume,map_location='cpu',weights_only=True,mmap=True)
    native=torch.load(export,map_location='cpu',weights_only=True,mmap=True)
    completed=json.loads((owner.REPORTS/'training-complete.json').read_text())
    if (completed['steps']!=3203 or completed['checkpoint_sha256']!=file_hash(export)
            or native['steps']!=3203 or native['planned_steps']!=3203
            or native['run_plan_sha256']!=file_hash(owner.REPORTS/'plan.json')
            or native['source_checkpoint_sha256']!=plan['base_checkpoint_sha256']
            or native['data_sha256']!=file_hash(owner.SOURCE/'admission.json')
            or set(native['model'])!=set(state['model'])
            or any(not torch.equal(t,native['model'][n]) for n,t in state['model'].items())):
        raise ValueError('Every actual native inference tensor/source/budget must equal the completed resume state')
    metrics=list(map(json.loads,(owner.REPORTS/'training-metrics.jsonl').read_text().splitlines()))
    if metrics!=state['metrics']:raise ValueError('Every actual optimizer metric must match the complete saved history')
    result['complete_source_and_rehearsal_exposures']=source_exposures(plan,state['metrics'],state['sampler_state'],complete=True)
    result.update(native_export_sha256=file_hash(export),all332_native_export_tensors_equal_complete_resume=True,
        historical_checkpoint_readback_artifact_preserved=True,
        original_reader_output_redirected_only=True,original_reader_source_sha256=file_hash(script),
        endpoint_wrapper_sha256=file_hash(__file__),exposure_auditor_sha256=file_hash(ROOT/'src/slm_lab/apps_epoch_evidence_v1.py'),
        actual_new_repair_training=False,model_capability_gain_established=False)
    atomic_json(target,result)
    print('complete_own706M_neural_state_export_and_all_source_exposures_reconciled',flush=True)


def native(plan):
    before=owner.REPORTS/'baseline';after=owner.REPORTS/'candidate'
    questions=list(map(json.loads,(ROOT/'data/knowledge-eval-v1/questions.jsonl').read_text().splitlines()))
    complete=json.loads((owner.REPORTS/'training-complete.json').read_text())
    results={};paths=[]
    for name in NAMES:
        a=json.loads((before/name).read_text());b=json.loads((after/name).read_text())
        if a['checkpoint_sha256']!=plan['base_checkpoint_sha256'] or b['checkpoint_sha256']!=complete['checkpoint_sha256']:
            raise ValueError('All actual native reports must bind the exact own pre/post epoch policies')
        matched(a,b,name)
        results[name]=dict(baseline=summarize(a,name,questions if name=='knowledge.json' else None),
            candidate=summarize(b,name,questions if name=='knowledge.json' else None),
            matched_actual_inputs_pixels_tokenizer_budget_precision_and_runtime=True)
        paths.extend([before/name,after/name])
    atomic_json(FOLDER/'matched-native-capability-comparison.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        assessments=results,baseline_checkpoint_sha256=plan['base_checkpoint_sha256'],
        candidate_checkpoint_sha256=complete['checkpoint_sha256'],
        report_files_sha256={str(p):file_hash(p) for p in paths},
        every_retained_generated_answer_or_worker_case_outcome_recounted=True,
        caption_retrieval_reported_without_retained_similarity_matrix=True,
        public_development_benchmarks_not_certified_independent_final_tests=True,
        math='Full original1319question matched CUDA/BF16 assessment is a separate checked queued stage.',
        paired_photo_full_source_and_fresh_human_alignment_still_required=True,
        new_repair_neural_updates=0,automatic_promotion=False,frontier_parity_established=False,goal_achieved=False))
    print('matched_native_code_knowledge_OCR_objects_caption_comparison_reconciled',flush=True)


def main():
    if QUEUE.exists() or FOLDER.exists():raise ValueError('Preserve earlier endpoint review; no implicit duplicate')
    plan=json.loads((owner.REPORTS/'plan.json').read_text());actor=plan['owner']
    files={str(owner.REPORTS/'plan.json'):file_hash(owner.REPORTS/'plan.json')}
    for p in [Path(__file__).resolve(),ROOT/'src/slm_lab/apps_epoch_evidence_v1.py',ROOT/'src/slm_lab/apps_native_metrics_v1.py',
            ROOT/'scripts/readback_apps_full_language_v1.py',ROOT/'tests/test_apps_epoch_evidence_v1.py',
            ROOT/'tests/test_apps_native_metrics_v1.py',ROOT/'reports/apps-epoch-evidence-v1-tests.log',
            ROOT/'reports/apps-native-metrics-v1-tests.log']:
        files[str(p)]=file_hash(p)
    for name in ['apps-epoch-evidence-v1-tests.log','apps-native-metrics-v1-tests.log']:
        if '2 passed' not in (ROOT/'reports'/name).read_text():raise ValueError('Actual full-draw and native output/fidelity checks required')
    guard=FileBindings(files)
    atomic_json(QUEUE,dict(queued_at=datetime.now(timezone.utc).isoformat(),actor=owner.resources.identity(__import__('os').getpid()),
        predecessor=actor,source_files_sha256=files,read_only=True,requires3203actual_updates=True,
        preserves_historical_checkpoint_readback=True,model_updates=0,goal_achieved=False))
    while not (owner.REPORTS/'training-complete.json').exists():
        if not owner.resources.same(actor):raise ValueError('Original full epoch ended without complete training evidence')
        guard.check();time.sleep(5)
    # Leave the CPUs available for native timed code/reference stress cases.
    # The full model/Adam/source scan runs only after those suites finish.
    while not all((owner.REPORTS/'candidate'/f'suite-{i}-complete.json').exists() for i in range(4)):
        if not owner.resources.same(actor):raise ValueError('Original native assessments ended without every complete suite')
        guard.check();time.sleep(5)
    guard.check(full=True);FileBindings(plan['source_files_sha256']).check(full=True)
    FOLDER.mkdir();checkpoint(plan);native(plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix-only',action='store_true');args=parser.parse_args()
    if args.prefix_only:prefix()
    else:main()
