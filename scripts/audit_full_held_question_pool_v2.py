"""CPU-only held-question exact-substring audit of explicit native input pools.

Scans complete current training presentations, both future human preference
responses, the queued code references, and three conservative older source
pools. No neural model is constructed and no generation or training occurs.
The older pools are possible exposure, not a claim that every token was used.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/full-held-question-audit-v2'
POOL = ROOT/'data/full-held-question-pool-v2'
LEGACY = ['wikipedia-expanded-v1', 'code-search-expanded-v1', 'broad-mixture-v1']


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('TOKENIZERS_PARALLELISM') != 'false' or OUT.exists():
        raise ValueError('Fresh CPU-hidden single-tokenizer-thread audit required')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from slm_lab.active_execution_input_bindings_v1 import input_bindings, merge_bindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources
    from slm_lab.held_question_scanner_v1 import QuestionScanner
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.model import ModelConfig
    from slm_lab.native_record_stream_v1 import NativeRecordStream
    from slm_lab.native_training_text_projection_v1 import NativeTrainingTextProjection
    from slm_lab.whole_human_data_v2 import WholeHumanData
    torch.set_num_threads(1)
    OUT.mkdir()
    started = time.monotonic(); stats = {}; scans = {}; progress_at = 0.
    pid = os.getpid(); proc = (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
    owner = dict(pid=pid,ticks=proc[19],CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false')
    atomic_json(OUT/'owner.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
        model_generations=0,model_updates=0,automatic_retry=False))
    try:
        rows = [json.loads(line) for line in (POOL/'question-registry.jsonl').open()]
        reserve = json.loads((POOL/'manifest.json').read_text())
        if (len(rows) != 17561 or any(not isinstance(r['question'],str) or not r['question'].strip() for r in rows)
                or any(any(m in r['question'] for m in ['<bos>','<eos>','<pad>']) for r in rows)):
            raise ValueError('Complete original registry without literal own-boundary-marker questions required')
        for name,digest in reserve['files_sha256'].items():
            if file_hash(POOL/name) != digest: raise ValueError('Held reserve bytes changed')
        phase_path = ROOT/'reports/phase-joint-v3/plan.json'
        phase = json.loads(phase_path.read_text())
        tokenizer_path = ROOT/'data/tokenizer-corpus/tokenizer.json'
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if (file_hash(tokenizer_path) != phase['tokenizer_sha256'] or tokenizer.get_vocab_size() != 32768
                or any(tokenizer.token_to_id(k) != v for k,v in [('<pad>',0),('<bos>',1),('<eos>',2)])):
            raise ValueError('Exact current own native tokenizer required')
        # Only the current own initializer supplies architecture settings. Its
        # model tensors are memory-mapped and never instantiated as a network.
        initial_path = Path(phase['initial_dense_settings']['initializer'])
        initial_digest = phase['initial_dense_settings']['initializer_sha256']
        if file_hash(initial_path) != initial_digest: raise ValueError('Own architecture source changed')
        raw = torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)
        config = ModelConfig(**raw['config']); del raw
        data = ExpandedJointSources(ROOT,config); projection = NativeTrainingTextProjection(data)
        human = WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
        code = ExpandedExecutionCorpus(ROOT)
        if human.manifest['tokenizer_sha256'] != phase['tokenizer_sha256'] or file_hash(code.tokenizer_path) != phase['tokenizer_sha256']:
            raise ValueError('All declared future optimization inputs must use the identical own tokenizer')
        bindings = dict(phase['source_files_sha256'])
        merge_bindings(bindings,reserve['source_files_sha256'])
        merge_bindings(bindings,input_bindings(ROOT,code))
        for name,digest in human.manifest['files_sha256'].items():
            if name.startswith('preferences.train'):
                merge_bindings(bindings,{str(human.root/name):digest})
        seeds = [Path(__file__),ROOT/'tests/test_native_record_stream_v1.py',ROOT/'tests/test_native_training_text_projection_v1.py',
            ROOT/'tests/test_held_question_scanner_v1.py']
        direct = [phase_path,tokenizer_path,POOL/'manifest.json',POOL/'question-registry.jsonl',
            human.root/'manifest.json',human.root/'admission.json',human.root/'independent-source-readback.json']
        direct += [POOL/name for name in reserve['files_sha256']]
        direct += python_sources(ROOT,seeds)
        direct += [ROOT/'reports'/name for name in ['native-record-stream-v1-tests.log',
            'native-training-text-projection-v1-tests.log','held-question-scanner-v1-tests.log']]
        merge_bindings(bindings,{str(p):file_hash(p) for p in direct})
        # Explicit raw pool manifests and their train shards only. This does not
        # recursively crawl unrelated historical neural state/checkpoint files.
        legacy = {}
        for name in LEGACY:
            folder = ROOT/'data'/name; path = folder/'corpus.json'; metadata = json.loads(path.read_text())
            if metadata['dtype'] != '<u2' or metadata['tokenizer_sha256'] != phase['tokenizer_sha256']:
                raise ValueError('Matching native raw train pools required')
            split = metadata['splits']['train']; legacy[name] = split
            if sum(r['tokens'] for r in split['shards']) != split['tokens']:
                raise ValueError('Every raw train shard must reconstruct its declared complete token count')
            merge_bindings(bindings,{str(path):file_hash(path)})
            merge_bindings(bindings,{str(folder/r['path']):r['sha256'] for r in split['shards']})
        # All actual runtime inputs are checked before the scan. Architecture
        # source is recorded separately to avoid re-reading its neural tensors.
        for name,digest in bindings.items():
            path = Path(name)
            if file_hash(path) != digest: raise ValueError('Audit runtime input differs: '+name)
            st = path.stat(); stats[name] = (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        expected_current = {k:sum(r[k] for r in phase['complete_source_budget'].values())
            for k in ['examples','complete_context_tokens','response_EOS_targets']}
        if expected_current != dict(examples=669863,complete_context_tokens=577662698,response_EOS_targets=467581587):
            raise ValueError('Exact existing full current curriculum required')
        plan = dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
            question_counts=dict(Counter(r['benchmark'] for r in rows)),current_native_expected=expected_current,
            future_human_expected=human.manifest['splits']['preferences.train'],
            future_code_original_sources=len(code),older_possible_source_tokens={k:v['tokens'] for k,v in legacy.items()},
            normalization='NFKC, Unicode casefold, collapse whitespace; exact original question substring',
            native_record_boundaries='BOS/EOS/PAD prevent matches crossing unrelated examples; raw records reconstructed across manifest-ordered shards',
            maximum_complete_raw_record_tokens=2_000_000,
            inspected_pool_order=['current_native','future_human','future_code']+LEGACY,
            architecture_source=dict(path=str(initial_path),sha256=initial_digest,only_config_read=True),
            source_files_sha256=bindings,no_reference_CoT_used=True,model_generations=0,model_updates=0,
            older_source_pool_is_potential_exposure_not_exact_consumption=True,
            complete_historical_ancestry_semantic_independence_and_unannotated_image_text_not_certified=True,
            automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(OUT/'plan.json',plan)
        def progress(scope,counts,force=False):
            nonlocal progress_at
            now = time.monotonic()
            if force or now-progress_at >= 15:
                value = dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,scope=scope,
                    counts=counts,elapsed_seconds=now-started,completed_scopes=list(scans),
                    no_exact_match_result_is_not_a_semantic_independence_certificate=True,goal_achieved=False)
                atomic_json(OUT/'progress.json',value);print(json.dumps(value),flush=True);progress_at=now
        def finish(scope,scanner,counts):
            result = dict(scope=scope,counts=counts,matched_original_question_ids=len(scanner.matches),
                matches=scanner.matches,all_declared_scope_inputs_scanned=True,
                question_counts_by_benchmark=dict(Counter(r['benchmark'] for r in rows if r['audit_id'] in scanner.matches)))
            atomic_json(OUT/(scope+'.json'),result);scans[scope]=result
            progress(scope,counts,True)
        scanner = QuestionScanner(rows); counts = dict(records=0,input_tokens=0,response_or_next_token_targets=0)
        families = {}
        for family in data.populations:
            local = dict(records=0,input_tokens=0,response_or_next_token_targets=0)
            for position,ids,labels in projection.family(family):
                scanner.native(tokenizer,ids,dict(scope='current_native',family=family,position=position))
                local['records']+=1;local['input_tokens']+=len(ids);local['response_or_next_token_targets']+=sum(v != -100 for v in labels[1:])
                if position % 1024 == 0:progress('current_native',dict(totals=counts,family=family,family_counts=local))
            original = phase['complete_source_budget'][family]
            if list(local.values()) != [original['examples'],original['complete_context_tokens'],original['response_EOS_targets']]:
                raise ValueError('Every current family must be fully scanned with its exact original native input/target budget')
            families[family] = local
            for k in counts:counts[k]+=local[k]
        finish('current_native',scanner,dict(totals=counts,families=families))
        scanner = QuestionScanner(rows); counts = dict(original_pairs=0,response_presentations=0,input_tokens=0,response_EOS_targets=0)
        for position in range(len(human)):
            for field in human.fields:
                ids,labels = human.sequence(position,field)
                scanner.native(tokenizer,ids,dict(scope='future_human',pair=position,field=field))
                counts['response_presentations']+=1;counts['input_tokens']+=len(ids);counts['response_EOS_targets']+=sum(v != -100 for v in labels[1:])
            counts['original_pairs']+=1
            if position % 1024 == 0:progress('future_human',counts)
        expected = human.manifest['splits']['preferences.train']
        if (counts['original_pairs'] != 85786 or counts['input_tokens'] != expected['packed_context_tokens']
                or counts['response_EOS_targets'] != expected['response_EOS_targets']):raise ValueError('All whole human responses required')
        finish('future_human',scanner,counts)
        scanner = QuestionScanner(rows); counts = dict(records=0,input_tokens=0,response_EOS_targets=0)
        for position in range(len(code)):
            ids,labels = code.sequence(position)
            scanner.native(tokenizer,ids,dict(scope='future_code',id=code.row(position)['id'],position=position))
            counts['records']+=1;counts['input_tokens']+=len(ids);counts['response_EOS_targets']+=sum(v != -100 for v in labels[1:])
            if position % 256 == 0:progress('future_code',counts)
        if counts != dict(records=3721,input_tokens=2287522,response_EOS_targets=584218):raise ValueError('Every original queued code reference required')
        finish('future_code',scanner,counts)
        for name,split in legacy.items():
            scanner = QuestionScanner(rows); stream = NativeRecordStream(); completed_shards = 0
            for shard in split['shards']:
                path = ROOT/'data'/name/shard['path']
                if path.stat().st_size != 2*shard['tokens']:raise ValueError('Every complete original raw train shard byte required')
                values = np.memmap(path,dtype='<u2',mode='r')
                for ids,record in stream.feed(values,dict(shard=shard['path'])):
                    scanner.native(tokenizer,ids,dict(scope=name,**record))
                del values;completed_shards+=1
                progress(name,dict(completed_shards=completed_shards,input_tokens=stream.input_tokens,records=stream.records))
            for ids,record in stream.finish():scanner.native(tokenizer,ids,dict(scope=name,**record))
            if stream.input_tokens != split['tokens']:raise ValueError('Every declared older possible-exposure token required')
            finish(name,scanner,dict(shards=completed_shards,input_tokens=stream.input_tokens,
                hard_boundary_tokens=stream.boundary_tokens,decoded_non_boundary_tokens=stream.output_tokens,records=stream.records))
        # Verify all bound inputs still have the identical bytes at completion;
        # stat checks catch replacement even when a same-SHA file was installed.
        for name,digest in bindings.items():
            st = Path(name).stat()
            if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns) != stats[name] or file_hash(Path(name)) != digest:
                raise ValueError('Audit runtime input changed while scanning: '+name)
        by_question=[]
        for row in rows:
            matches={scope:value['matches'][row['audit_id']] for scope,value in scans.items() if row['audit_id'] in value['matches']}
            by_question.append(dict(audit_id=row['audit_id'],benchmark=row['benchmark'],id=row['id'],
                previously_sampled_for_local_evaluation=row['previously_sampled_for_local_evaluation'],
                inspected_exact_match_scopes=list(matches),matches=matches,
                exact_question_found_in_inspected_inputs=bool(matches)))
        report_path=OUT/'question-results.jsonl'
        report_path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in by_question))
        fresh=[r for r in by_question if r['benchmark']=='MMLU-Pro' and not r['previously_sampled_for_local_evaluation']
            and not r['exact_question_found_in_inspected_inputs']]
        fresh_ids={r['id'] for r in fresh}; original_rows=map(json.loads,(POOL/'mmlu-pro.original-test.jsonl').open())
        fresh_path=OUT/'mmlu-pro.fresh-no-exact-match-in-inspected-pools.jsonl'
        fresh_path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in original_rows if r['id'] in fresh_ids))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
            plan_sha256=file_hash(OUT/'plan.json'),elapsed_seconds=time.monotonic()-started,
            all_six_declared_scopes_completed=True,scope_counts={k:v['counts'] for k,v in scans.items()},
            matched_original_question_counts={k:v['question_counts_by_benchmark'] for k,v in scans.items()},
            question_results_sha256=file_hash(report_path),fresh_MMLU_original_test_questions=len(fresh),
            fresh_MMLU_cohort_sha256=file_hash(fresh_path),
            final_files_sha256={p.name:file_hash(p) for p in OUT.glob('*.json') if p.name not in ['progress.json','owner.json']},
            fresh_means_not_previously_sampled_locally_and_no_exact_question_match_in_inspected_inputs=True,
            complete_historical_ancestry_semantic_independence_and_unannotated_image_text_not_certified=True,
            model_generations=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),fresh_MMLU_questions=len(fresh))),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
            type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            complete=False,automatic_retry=False,goal_achieved=False))
        raise


if __name__ == '__main__': main()
