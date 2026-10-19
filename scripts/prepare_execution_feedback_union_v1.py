"""Materialize and reconcile the entire admitted Java/Python execution union."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_code_feedback_v1 import sequence
from slm_lab.execution_collective_v2 import feedback_identity
from slm_lab.execution_feedback_reader_v1 import ExecutionFeedback
from slm_lab.execution_feedback_union_v1 import ADMISSION_FORMAT,FORMAT,ExecutionFeedbackUnion,merge_files
from slm_lab.execution_preference_v1 import validate
from slm_lab.python_feedback_reader_v2 import PythonExecutionFeedbackV2

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/execution-feedback-union-v1'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'


def main():
    if DATA.exists():raise ValueError('New owned complete union required; no overwrite or implicit resume')
    tokenizer=Tokenizer.from_file(str(TOKEN))
    owners=[ExecutionFeedback(ROOT/'data/execution-code-feedback-v2',tokenizer,TOKEN),
        PythonExecutionFeedbackV2(ROOT/'data/python-execution-feedback-v2',tokenizer,TOKEN)]
    files={};metadata={}
    for name,owner in zip(['Java','Python'],owners):
        owner.guard.check(full=True)
        merge_files(files,{str(p):h for p,h in owner.guard.files.items()})
        metadata[name]=dict(folder=str(owner.folder),manifest_sha256=file_hash(owner.folder/'manifest.json'),
            admission_sha256=file_hash(owner.folder/'admission.json'))
    merge_files(files,{str(Path(__file__).resolve()):file_hash(__file__),
        str(ROOT/'src/slm_lab/execution_feedback_union_v1.py'):file_hash(ROOT/'src/slm_lab/execution_feedback_union_v1.py')})
    guard=FileBindings(files)
    pairs=[r for owner in owners for r in owner.pairs];positives=[r for owner in owners for r in owner.positives]
    if len(pairs)!=374 or len(positives)!=195:raise ValueError('Every210Java/164Python pair and109Java/86Python original must remain')
    held={r['group_id'] for r in map(json.loads,(ROOT/'data/exercism-support-context-v3/valid.jsonl').read_text().splitlines())}
    for values in [pairs,positives]:
        if len(values)!=len({r['id'] for r in values}) or any(r['split']!='train' or r['group_id'] in held or r['human_preferences'] is not False for r in values):
            raise ValueError('Original distinct train-only machine feedback source roles required')
    DATA.mkdir()
    for name,values in [('pairs.jsonl',pairs),('passing-originals.jsonl',positives)]:
        (DATA/name).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in values))
    manifest=dict(format=FORMAT,created_at=datetime.now(timezone.utc).isoformat(),parents=metadata,
        tokenizer_sha256=file_hash(TOKEN),finite_assertion_pairs=len(pairs),complete_passing_originals=len(positives),
        artifact_files_sha256={p.name:file_hash(p) for p in DATA.glob('*.jsonl')},source_files_sha256=files,
        per_language=dict(Java=dict(positives=109,pairs=210),Python=dict(positives=86,pairs=164)),
        original_code_and_all30_replay_populations_unchanged=True,training_admitted=False,human_preferences=False,
        model_updates=0,goal_achieved=False)
    atomic_json(DATA/'manifest.json',manifest)
    # Read back all rows independently of the writer; then reconstruct every
    # complete original own-BPE question/response/EOS and pair mask.
    actual_pairs=list(map(json.loads,(DATA/'pairs.jsonl').read_text().splitlines()))
    actual_positives=list(map(json.loads,(DATA/'passing-originals.jsonl').read_text().splitlines()))
    if actual_pairs!=pairs or actual_positives!=positives:raise ValueError('Exact complete union readback differs')
    examples=[];contexts={'positive':[],'chosen':[],'rejected':[]};targets={k:0 for k in contexts}
    for row in actual_positives:
        ids,labels=sequence(tokenizer,row['prompt'],row['response'])
        if (len(ids)!=row['context_tokens'] or sum(y!=-100 for y in labels)!=row['target_tokens']
                or hashlib.sha256(row['response'].encode()).hexdigest()!=row['source_response_sha256']):raise ValueError('Original complete positive source/mask differs')
        contexts['positive'].append(len(ids));targets['positive']+=sum(y!=-100 for y in labels)
    for row in actual_pairs:
        values={name:sequence(tokenizer,row['prompt'],row[name]) for name in ['chosen','rejected']}
        for name,(ids,labels) in values.items():
            if (len(ids)!=row[name+'_context_tokens'] or sum(y!=-100 for y in labels)!=row[name+'_target_tokens']
                    or hashlib.sha256(row[name].encode()).hexdigest()!=row[name+'_sha256']):raise ValueError('Exact complete original preference source/mask differs')
            contexts[name].append(len(ids));targets[name]+=sum(y!=-100 for y in labels)
        examples.append(dict(id=row['id'],split=row['split'],**values,
            feedback_source='machine_execution_of_original_publisher_tests',human_preferences=False,finite_execution_verified=True))
    validate(examples,True);guard.check(full=True)
    files={**files,str(DATA/'manifest.json'):file_hash(DATA/'manifest.json'),
        **{str(DATA/name):digest for name,digest in manifest['artifact_files_sha256'].items()}}
    admission=dict(format=ADMISSION_FORMAT,completed_at=datetime.now(timezone.utc).isoformat(),passed=True,
        candidate_manifest_sha256=file_hash(DATA/'manifest.json'),all_parent_views_retained_exactly=True,
        source_roles_only_train=True,machine_feedback_not_human_preferences=True,held_group_overlap=0,
        finite_assertion_pairs=374,complete_passing_originals=195,original_sources_accounted=195,
        per_language=manifest['per_language'],encoder_truncated_sequences=0,
        actual_maximum_context_tokens={k:max(v) for k,v in contexts.items()},full_response_EOS_target_tokens=targets,
        original_code_and_all30_replay_populations_unchanged=True,source_files_sha256=files,
        model_updates=0,goal_achieved=False,
        scope='Exact union of independently admitted finite publisher program/assertion evidence. No new correctness claim, human preferences, full-model updates or production readiness.')
    atomic_json(DATA/'admission.json',admission)
    loaded=ExecutionFeedbackUnion(DATA,tokenizer,TOKEN)
    identity=feedback_identity(loaded,file_hash(TOKEN))
    print(json.dumps(dict(admission=str(DATA/'admission.json'),identity=identity,
        maximum_context_tokens=admission['actual_maximum_context_tokens'],goal_achieved=False),indent=2))


if __name__=='__main__':main()
