"""Prepare lossless original/completion/execution-qualified repair train views.

Derived views reuse existing licensed public functions. They are not new raw
functions, new human feedback, held data, or evidence of a model quality gain.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/documented-doctest-verified-v1'
OUTPUT=ROOT/'data/verified-code-views-v1'
REPORTS=ROOT/'reports/verified-code-views-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('All GPUs must be hidden during data preparation')
    if OUTPUT.exists() or REPORTS.exists():raise ValueError('Preserve previous complete or partial preparations')
    from tokenizers import Tokenizer
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.codenet_token_pack_v1 import CodePackWriter,CompleteCodePack
    from slm_lab.code_training_views_v1 import base_row,completion_view,repair_candidates,encode_view,verify_mutation,original_program
    from slm_lab.public_example_code_reward_v1 import PublicExampleCodeReward
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    if '4 passed' not in (ROOT/'reports/code-training-views-v1-tests.log').read_text() or '1 passed' not in (ROOT/'reports/code-training-views-v1-mutation-tests.log').read_text():
        raise ValueError('Actual UTF8/roles/full-target and confined complete-example mutation checks required')
    if shutil.disk_usage(ROOT).free<27*1024**3:raise ValueError('Preserve persistent writer reserve')
    old=CompleteCodePack(SOURCE,'train','training');catalog=PublicExampleCodeReward(ROOT)
    with gzip.open(SOURCE/'original-rows.jsonl.gz','rt') as f:parents=list(map(json.loads,f))
    if len(parents)!=len(old) or any(r['token_pack_index']!=i for i,r in enumerate(parents)):raise ValueError('Every original source row/order required')
    paths=[Path(__file__),ROOT/'src/slm_lab/code_training_views_v1.py',ROOT/'src/slm_lab/documented_doctest_v1.py',
        ROOT/'src/slm_lab/public_example_code_reward_v1.py',ROOT/'tests/test_code_training_views_v1.py',
        ROOT/'reports/code-training-views-v1-tests.log',ROOT/'reports/code-training-views-v1-mutation-tests.log',
        SOURCE/'manifest.json',SOURCE/'admission.json',SOURCE/'publisher-license-notices.jsonl',
        ROOT/'reports/documented-doctest-verified-v1/independent-complete-source-and-runtime-readback-v1.json',
        ROOT/'data/documented-doctest-stage-v1/candidates.jsonl',ROOT/'reports/documented-doctest-verified-v1/observations.jsonl']
    paths.extend(SOURCE/name for name in old.manifest['files_sha256'])
    bindings={str(p):file_hash(p) for p in paths};guard=FileBindings(bindings);started=time.monotonic()
    OUTPUT.mkdir();REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bindings,
        original_unique_functions=len(parents),original_examples=715,views=['original_instruction','prefix_completion','verified_statement_repair'],
        maximum_mutants_per_function=4,complete_context_limit=4096,workers=2,
        repair_requires_completed_original_example_failure=True,training_only_original_roles=True,
        FIM_research_source='https://arxiv.org/abs/2207.14255v1',
        research_source_use='Data transformation only; no external model weights, teacher or benchmark solution loaded.',
        independent_reader_and_training_admission_pending=True,model_updates=0,goal_achieved=False))
    tokenizer=Tokenizer.from_file(str(SOURCE/'tokenizer.json'));jobs=[]
    for i,parent in enumerate(parents):
        original_program(parent);jobs.append((i,parent))
    def qualify(item):
        index,parent=item;key=parent['parent_source']['id'];attempts=[]
        for row in repair_candidates(parent):
            try:encode_view(row,tokenizer)
            except ValueError as error:
                attempts.append(dict(mutant_program_sha256=row['mutant_program_sha256'],status='complete_context_qualification',error=str(error)));continue
            result=verify_mutation(catalog.candidates[key],catalog.references[key],row['mutant_program'])
            attempts.append(result)
            if result['qualifies_as_observed_failure']:
                row['published_example_failure_gate_pending']=False
                row['published_example_failure_gate']=dict(complete_original_examples_observed=True,
                    fraction_passed=result['fraction_passed'],mutant_program_sha256=result['mutant_program_sha256'],
                    scored_original_examples=result['scored_original_examples'])
                return index,row,attempts
        return index,None,attempts
    qualifications={}
    with (REPORTS/'mutation-observations.jsonl').open('x') as log,ThreadPoolExecutor(max_workers=2) as pool:
        for index,row,attempts in pool.map(qualify,jobs):
            qualifications[index]=row
            log.write(json.dumps(dict(source_position=index,source_id=parents[index]['parent_source']['id'],
                repair_view_qualified=row is not None,attempts=attempts),ensure_ascii=False)+'\n');log.flush()
            if len(qualifications)%25==0:
                atomic_json(REPORTS/'status.json',dict(completed_functions=len(qualifications),total_functions=len(parents),
                    qualified_repair_views=sum(r is not None for r in qualifications.values()),model_updates=0,goal_achieved=False))
    for name in ['tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-license-notices.jsonl']:
        shutil.copyfile(SOURCE/name,OUTPUT/name)
    (OUTPUT/'publisher-README.md').write_text('Original publisher-train public Python functions with retained per-source notices.\n'
        'Original instruction views retain exact former input IDs and complete formatted targets.\n'
        'Completion and statement-repair views derive framing/targets from exact original source slices.\n'
        'Reassembling their prefix/continuation or prefix/middle/suffix reproduces every original program byte.\n'
        'A repair view requires a deterministic pass-substitution mutant to fail the complete original published examples under confinement.\n'
        'Mutants are machine derivatives, not publisher programs, own-policy outputs, new human labels or generally incorrect programs.\n'
        'These are additional presentations of the same functions, not new unique raw sources. No held source is promoted to training.\n'
        'Finite published examples do not prove general correctness/efficiency or new model capability. Full training admission awaits a separate reader.\n')
    writer=CodePackWriter(OUTPUT,maximum_bytes=64*1024**2,minimum_free_gib=26);counts=Counter();dispositions=[]
    try:
        for i,parent in enumerate(parents):
            full=base_row(parent,'original_instruction',parent['prompt'],parent['response'],native_framing='retained_original_tokens',
                original_source_position=i,complete_original_formatted_target_unchanged=True,complete_original_function_bytes_preserved=True)
            ids,labels=old.sequence(i);writer.add(full,ids,labels);counts[full['view']]+=1
            modes=[completion_view(parent)]
            if qualifications[i] is not None:modes.append(qualifications[i])
            for row in modes:
                ids,labels=encode_view(row,tokenizer);row['original_source_position']=i
                writer.add(row,ids,labels);counts[row['view']]+=1
            dispositions.append(dict(original_source_position=i,original_source_id=parent['parent_source']['id'],
                original_instruction_retained=True,completion_retained=True,repair_retained=qualifications[i] is not None,
                missing_repair_is_explicit_qualification=True))
        atomic_json(OUTPUT/'view-dispositions.json',dispositions)
        guard.check(full=True)
        manifest=writer.finish(dict(original_source_manifest_sha256=file_hash(SOURCE/'manifest.json'),
            original_admission_sha256=file_hash(SOURCE/'admission.json'),source_files_sha256=bindings,
            views=dict(counts),semantic_unique_functions=len(parents),new_unique_raw_functions=0,
            notice_sha256=file_hash(OUTPUT/'publisher-license-notices.jsonl'),
            view_dispositions_sha256=file_hash(OUTPUT/'view-dispositions.json'),
            mutation_observations_sha256=file_hash(REPORTS/'mutation-observations.jsonl'),
            FIM_research='https://arxiv.org/abs/2207.14255v1',training_admission_requires_separate_complete_reader=True))
        writer=None
        atomic_json(REPORTS/'prepared.json',dict(at=datetime.now(timezone.utc).isoformat(),views=dict(counts),unique_original_functions=len(parents),
            new_raw_unique_functions=0,counts=manifest['counts'],maximum_context_limit=4096,
            supplied_notices_retained=94,independent_reader_pending=True,admitted_to_training=False,
            model_updates=0,human_preference_labels=0,seconds=time.monotonic()-started,goal_achieved=False))
        print(json.dumps(dict(views=dict(counts),counts=manifest['counts'],seconds=time.monotonic()-started)),flush=True)
    finally:
        if writer is not None:writer.close()


if __name__=='__main__':main()
