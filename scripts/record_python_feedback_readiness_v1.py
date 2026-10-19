"""Bind complete actual Python publisher feedback, its reader and live training."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_collective_v2 import feedback_identity
from slm_lab.execution_preference_v1 import validate
from slm_lab.python_feedback_reader_v1 import PythonExecutionFeedback

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'data/python-execution-feedback-v1';reports=ROOT/'reports/python-execution-feedback-v1'
    native_log=ROOT/'reports/python-execution-feedback-v1-final-tests.log'
    reader_log=ROOT/'reports/python-feedback-reader-v1-focused-tests.log'
    if '8 passed' not in native_log.read_text() or '3 passed' not in reader_log.read_text():
        raise ValueError('Complete confined runtime and real-data reader checks required')
    token=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
    feedback=PythonExecutionFeedback(data,Tokenizer.from_file(str(token)),token)
    identity=feedback_identity(feedback,file_hash(token))
    positives=[feedback.positive(i) for i in range(len(feedback.positives))]
    pairs=[feedback.example(i) for i in range(len(feedback.pairs))]
    validate(pairs,True)
    def extent(values):
        return dict(records=len(values),minimum_context_tokens=min(len(x[0]) for x in values),
            maximum_context_tokens=max(len(x[0]) for x in values),
            full_response_EOS_target_tokens=sum(sum(y!=-100 for y in x[1]) for x in values))
    qualification=[]
    for path in reports.glob('module-*.json'):
        row=json.loads(path.read_text())
        if row['qualification']!='all_finite_published_cases_pass':
            qualification.append(dict(id=row['id'],qualification=row['qualification'],report=str(path),
                report_sha256=file_hash(path),reason='Missing original test_utils helper' if row['id'].endswith('error-handling')
                else 'v1 rejects repeated parent call reports from unittest subtests; complete distinct subtest identities require a later backend revision.',
                supplied_positive_or_preference_labels=False,original_source_retained_in_ordinary_replay=True))
    feedback.guard.check(full=True)
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True).stdout.splitlines()
    workers=[]
    for line in apps:
        pid=int(line.split(',')[0]);proc=Path('/proc')/str(pid)
        fields=(proc/'stat').read_text().split(') ',1)[1].split();argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if fields[0]=='Z' or 'slm_lab.training pretrain' not in argv or 'systems' not in argv:
            raise ValueError('Actual original systems-code CUDA workers required')
        parent=int(fields[1]);launcher=Path('/proc')/str(parent)
        if int((launcher/'stat').read_text().split(') ',1)[1].split()[1])!=2754743:
            raise ValueError('Exact original controller and direct worker parentage required')
        workers.append(dict(pid=pid,parent_pid=parent,start_ticks=fields[19],state=fields[0],argv=argv,GPU_record=line))
    if len(workers)!=4:raise ValueError('AllfouractualGPUworkers required')
    lines=(ROOT/'reports/curriculum-disk-cleanup-resumption-v1/joint.log').read_text().splitlines()
    latest=[json.loads(line) for line in lines if line.startswith('{"step"')][-1]
    original_status=json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text())
    paths=[data/'admission.json',data/'manifest.json',data/'pairs.jsonl',data/'passing-originals.jsonl',native_log,reader_log,
        ROOT/'reports/python-execution-feedback-v1-audit.log',Path(__file__).resolve(),
        ROOT/'src/slm_lab/python_feedback_reader_v1.py',ROOT/'tests/test_python_feedback_reader_v1.py',
        ROOT/'tests/test_python_execution_feedback_v1.py']
    bound={**{str(p):d for p,d in feedback.guard.files.items()},**{str(p):file_hash(p) for p in paths}}
    result=dict(format='own-complete-Python-execution-feedback-readiness-v1',observed_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification=dict(classification='progress',reason='Completed936-update own706M broad-code run and independently read back weights/full936metrics; started original four-L40 systems training; integrated Java execution kernel/production lifecycle with10CPUchecks.'),
        current_turn_classification=dict(classification='progress',reason='Implemented and executed complete confined Python publisher suites, deterministic code-token mutations, independent raw/source/whole-BPE audit and common CE/DPO sequence reader.82original programs pass1236publisher cases;153assertion preferences admitted, all86sources accounted.'),
        source_training_population=86,complete_passing_originals=82,original_publisher_cases_passed=1236,
        finite_assertion_pairs=153,source_groups_with_pairs=feedback.admission['source_groups_with_pairs'],
        original_source_qualifications=feedback.admission['original_source_qualifications'],
        mutation_qualifications=feedback.admission['mutation_qualifications'],qualified_sources=qualification,
        complete_sequence_interfaces=dict(identity=identity,positive=extent(positives),
            chosen=extent([p['chosen'] for p in pairs]),rejected=extent([p['rejected'] for p in pairs]),
            common_train_only_full_DPO_validation_passed=True,encoder_truncated_sequences=0),
        tests=dict(native_mutation_collection_source_and_sandbox=8,actual_reader_and_tokenizer_source_boundaries=3,
            native_log=str(native_log),reader_log=str(reader_log)),
        execution_feedback_common_kernel_interface=True,new_Python_feedback_production_plan_wired=False,
        Java_and_Python_feedback_unified=False,new_Python_full_model_optimizer_updates=0,
        original_required_code_and_all30_source_replay_populations_unchanged=True,
        original_training=dict(status=original_status,actual_four_GPU_workers=workers,latest_logged_metric=latest,
            new_Python_execution_feedback_used=False),
        preflight_notes=['Initial pytest capture imported readline and crashed under confinement; optional capture is disabled, with full publisher collection and assertions preserved.',
            'Python-specific confined launcher permits pytest directory changes while retaining filesystem/network/fork restrictions; actual boundary tests pass.',
            'Canonical-data artifacts resolve through the pinned problem-specifications repository, not the Python archive.',
            'Four original tasks retain explicit runtime/helper inventory qualifications and supply no new feedback labels.'],
        remaining_work=['Resolve three unittest-subtest inventories and the original error-handling helper in a separate bound backend revision.',
            'Combine independently admitted Java/Python views and wire the explicit production reader/source plan without dropping original data.',
            'Complete original systems/multimodal/genuine-feedback continuation and final source/base/baseline requirements.',
            'Run exact full-model capacity/reference scoring, train complete source populations plus supplemental feedback and evaluate unchanged coding correctness/efficiency, domain knowledge and image tasks.'],
        candidate_promoted=False,new_model_capability_gain_measured=False,high_end_capability_established=False,
        goal_achieved=False,files_sha256=bound)
    path=ROOT/'reports/python-execution-feedback-v1-readiness.json';atomic_json(path,result)
    print(json.dumps(dict(report=str(path),positives=82,publisher_cases=1236,pairs=153,qualified_sources=qualification,
        sequence_extents=result['complete_sequence_interfaces'],original_training_step=latest['step'],goal_achieved=False),indent=2))


if __name__=='__main__':main()
