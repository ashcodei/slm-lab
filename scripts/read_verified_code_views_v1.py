"""Separate byte/mask/source reader and confined negative replay before admission."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Reader must hide every GPU')
    from tokenizers import Tokenizer
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.code_quality import instruction_code
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.documented_doctest_v1 import execute,contract,compare
    from slm_lab.public_example_code_reward_v1 import PublicExampleCodeReward
    folder=ROOT/'data/verified-code-views-v1';old_folder=ROOT/'data/documented-doctest-verified-v1'
    report=ROOT/'reports/verified-code-views-v1/independent-reader-v1.json'
    if report.exists() or (folder/'admission.json').exists():raise ValueError('Preserve previous readback/admission')
    pack=CompleteCodePack(folder,'train','audit');old=CompleteCodePack(old_folder,'train','training')
    with gzip.open(folder/'original-rows.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
    with gzip.open(old_folder/'original-rows.jsonl.gz','rt') as f:parents=list(map(json.loads,f))
    catalog=PublicExampleCodeReward(ROOT);tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
    files=[Path(__file__),ROOT/'src/slm_lab/documented_doctest_v1.py',folder/'manifest.json',old_folder/'manifest.json',old_folder/'admission.json']
    files.extend(folder/name for name in pack.manifest['files_sha256'])
    files.extend(old_folder/name for name in old.manifest['files_sha256'])
    provenance=pack.manifest['provenance']
    for name,key in [('publisher-license-notices.jsonl','notice_sha256'),('view-dispositions.json','view_dispositions_sha256')]:
        if file_hash(folder/name)!=provenance[key]:raise ValueError('Extra retained source artifact changed')
        files.append(folder/name)
    mutation_log=ROOT/'reports/verified-code-views-v1/mutation-observations.jsonl'
    if file_hash(mutation_log)!=provenance['mutation_observations_sha256']:raise ValueError('Actual negative ledger changed')
    files.append(mutation_log);bindings={str(p):file_hash(p) for p in files};guard=FileBindings(bindings)
    if (folder/'publisher-license-notices.jsonl').read_bytes()!=(old_folder/'publisher-license-notices.jsonl').read_bytes():
        raise ValueError('Every provided notice byte must remain exact')
    started=time.monotonic();counts=Counter();by_parent={i:set() for i in range(len(parents))};negatives=[];tokens=targets=0
    def sha(s):return hashlib.sha256(s.encode()).hexdigest()
    for i,row in enumerate(rows):
        position=row['original_source_position'];parent=parents[position];mode=row['view'];key=parent['parent_source']['id']
        if (row['token_pack_index']!=i or mode in by_parent[position] or row['split']!='train' or row['publisher_split']!='train'
                or row['parent_source']!=parent['parent_source'] or row['parent_source']['original_partition']!='train'
                or row['parent_training_row_id']!=parent['id'] or row['group_id']!=parent['group_id']):raise ValueError('Exact original role/identity/view required')
        ids,labels=pack.sequence(i);start=next(j for j,x in enumerate(labels) if x!=-100)
        if (ids[0]!=1 or ids[-1]!=2 or len(ids)>4096 or labels[:start]!=[-100]*start or labels[start:]!=ids[start:]
                or sha(parent['prompt'])!=row['original_parent_prompt_sha256'] or sha(parent['response'])!=row['original_formatted_target_sha256']):
            raise ValueError('Whole exact input/response/EOS required')
        program=instruction_code(parent['response']);raw=program.encode()
        if sha(program)!=row['original_complete_program_sha256']:raise ValueError('Original program differs')
        if mode=='original_instruction':
            if row['prompt']!=parent['prompt'] or row['response']!=parent['response'] or (ids,labels)!=old.sequence(position):
                raise ValueError('Every original full-target ID and label must be unchanged')
        else:
            expected_prefix=row['prompt'] if mode=='prefix_completion' else 'User: '+row['prompt']+'\n\nAssistant:'
            if (ids!=[1,*tokenizer.encode(expected_prefix).ids,*tokenizer.encode(row['response']).ids,2]
                    or tokenizer.decode(ids[1:start])!=expected_prefix or tokenizer.decode(ids[start:-1])!=row['response']):
                raise ValueError('Independent complete native text/target reconstruction differs')
            if mode=='prefix_completion':
                k=row['source_split_byte']
                if (raw[:k].decode()!=row['original_prefix'] or raw[k:].decode()!=row['original_continuation']
                        or row['response']!=raw[k:].decode()+'\n```' or not row['prompt'].endswith('```python\n'+raw[:k].decode())):
                    raise ValueError('Exact full source prefix/continuation bytes required')
            elif mode=='verified_statement_repair':
                a,b=row['source_span_bytes'];prefix=raw[:a].decode();middle=raw[a:b].decode();suffix=raw[b:].decode()
                if (row['original_prefix']!=prefix or row['original_middle']!=middle or row['original_suffix']!=suffix
                        or row['response']!=middle or row['mutant_program']!=prefix+'pass'+suffix
                        or prefix+row['placeholder']+suffix not in row['prompt'] or row['published_example_failure_gate_pending']):
                    raise ValueError('Exact whole-source repair/target bytes required')
                negatives.append((i,key,row['mutant_program']))
            else:raise ValueError('Unknown derived presentation')
            if row['reconstructed_program_sha256']!=sha(program):raise ValueError('Exact reconstructed original function required')
        tokens+=len(ids);targets+=len(ids)-start;counts[mode]+=1;by_parent[position].add(mode)
    dispositions=json.loads((folder/'view-dispositions.json').read_text())
    if len(dispositions)!=len(parents) or len(rows)!=len(pack):raise ValueError('Every original disposition and derived row required')
    for i,d in enumerate(dispositions):
        expected={'original_instruction','prefix_completion'}|({'verified_statement_repair'} if d['repair_retained'] else set())
        if (d['original_source_position']!=i or d['original_source_id']!=parents[i]['parent_source']['id'] or by_parent[i]!=expected):
            raise ValueError('No source silently omitted from complete views')
    raw_path=report.with_name('independent-negative-replays-v1.jsonl')
    if raw_path.exists():raise ValueError('Preserve earlier replay attempts')
    def replay(item):
        import copy
        i,key,mutant=item;candidate=copy.deepcopy(catalog.candidates[key]);formatted='```python\n'+mutant+'\n```'
        candidate['source']=dict(candidate['source'],id='reader-machine-mutation:'+key,response=formatted,annotation_origin='Independent deterministic mutation replay; no model generation')
        candidate.update(program=mutant,program_sha256=sha(mutant),existing_formatted_source_target_sha256=sha(formatted))
        actual=execute(candidate,seconds=5);examples=contract(catalog.candidates[key]);flags=[compare(e,r) for e,r in zip(examples,actual.get('raw_results',[]))]
        if not actual.get('all_original_examples_observed') or len(flags)!=len(examples) or flags!=actual['example_passes'] or all(flags):
            raise ValueError('Every repair negative must independently fail an actual original example')
        return dict(view_position=i,original_source_id=key,mutant_program_sha256=sha(mutant),raw_execution=actual)
    with raw_path.open('x') as f,ThreadPoolExecutor(max_workers=2) as pool:
        for observation in pool.map(replay,negatives):f.write(json.dumps(observation,ensure_ascii=False)+'\n')
    expected=dict(examples=len(rows),context_tokens=tokens,target_tokens=targets)
    if expected!=pack.manifest['counts']['train'] or dict(counts)!=provenance['views']:raise ValueError('Every actual view/count must reconstruct')
    guard.check(full=True)
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),dataset_manifest_sha256=file_hash(folder/'manifest.json'),
        all_rows_ids_masks_roles_and_source_bytes_reconstructed=True,counts=expected,views=dict(counts),original_functions=len(parents),
        exact_unchanged_full_target_rows=counts['original_instruction'],negative_programs_independently_reexecuted=len(negatives),
        original_reference_examples_previously_independently_passed=715,provided_notice_bytes_unchanged=True,
        negative_replays_sha256=file_hash(raw_path),source_files_sha256=bindings,seconds=time.monotonic()-started,
        new_unique_raw_functions=0,neural_updates=0,quality_improvement_proven=False,goal_achieved=False)
    atomic_json(report,result)
    atomic_json(folder/'admission.json',dict(dataset_manifest_sha256=result['dataset_manifest_sha256'],admitted_to_training=True,
        independent_reader=str(report),independent_reader_sha256=file_hash(report),unique_original_functions=len(parents),
        derived_training_rows=len(rows),new_unique_raw_functions=0,human_preferences=0,neural_updates=0,goal_achieved=False))
    CompleteCodePack(folder,'train','training');print(json.dumps(dict(admitted=True,views=dict(counts),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
