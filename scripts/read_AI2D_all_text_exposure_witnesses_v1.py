"""Independently rederive every recorded AI2D text-exposure source witness.

Native current matches use actual training batches, without the audit's token
projection. Older source records are read whole across manifest-ordered shard
bounds before decoding. No substring automaton, NN or model generation is used.
This verifies reported witnesses and accounting, not an independent negative
rescan of3.25B tokens or proof of semantic/image/complete ancestral independence.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import unicodedata

import numpy as np
import torch

from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.native_raw_witness_v1 import reconstruct
from slm_lab.whole_human_data_v2 import WholeHumanData

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/AI2D-full-text-exposure-audit-v1'
OUT = ROOT / 'reports/AI2D-all-text-exposure-witnesses-v1-readback'
SCOPES = ('current_native', 'future_human', 'future_code', 'wikipedia-expanded-v1', 'code-search-expanded-v1', 'broad-mixture-v1')
KINDS = ('original_question', 'question_with_ordered_options', 'whole_native_prompt')


def live(identity):
    try: fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return False
    return fields[19] == identity['ticks'] and fields[0] not in ['Z', 'X']


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden all-witness readback required')
    torch.set_num_threads(1); OUT.mkdir(); expires = time.time() + 24 * 3600
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    source_plan_path = RUN / 'plan.json'; source_plan = json.loads(source_plan_path.read_text())
    producer = source_plan['owner']; population = FullAI2D(ROOT); tokenizer = population.tokenizer
    if source_plan['original_population_sha256'] != population.identity_sha256: raise ValueError('Exact original complete AI2D population required')
    phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
    architecture = source_plan['architecture_source']
    if file_hash(architecture['path']) != architecture['sha256']: raise ValueError('Original own architecture configuration source required')
    raw = torch.load(architecture['path'], map_location='cpu', weights_only=True, mmap=True)
    config = ModelConfig(**raw['config']); del raw
    data = ExpandedJointSources(ROOT, config)
    human = WholeHumanData(ROOT / 'data/whole-human-alignment-v2', 'preferences', 'train', 'optimization')
    code = ExpandedExecutionCorpus(ROOT)
    files = dict(source_plan['source_files_sha256'])
    extra = [source_plan_path, phase_path, ROOT / 'tests/test_native_raw_witness_v1.py', ROOT / 'reports/native-raw-witness-v1-tests.log',
        *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_native_raw_witness_v1.py'])]
    files.update({str(p): file_hash(p) for p in extra}); guard = FileBindings(files)
    questions = {}
    for case in population.cases:
        variants = dict(original_question=case['question'], question_with_ordered_options=case['question'] + '\n' +
            '\n'.join(chr(65 + index) + '. ' + option for index, option in enumerate(case['options'])),
            whole_native_prompt=case['whole_native_prompt'])
        for kind, text in variants.items(): questions[case['id'] + ':' + kind] = dict(case=case, kind=kind, text=text)
    if len(questions) != 9264: raise ValueError('Every original question and all three literal signatures required')
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, exact_producer=producer,
        scopes=list(SCOPES), original_population_sha256=population.identity_sha256, original_questions=3088, literal_signatures=9264,
        source_files_sha256=files, expires_at_unix=expires, producer_projection_and_automaton_not_used=True,
        independent_full_negative_rescan_not_performed=True, neural_generation_or_training_updates=0, goal_achieved=False))
    scopes = {}; artifacts = [OUT / 'plan.json']; context_cache = {}; witness_rows = []
    try:
        def await_file(path):
            while not path.is_file():
                if not live(producer) or (RUN / 'failure.json').exists(): raise ValueError('Exact source auditor ended without required immutable output')
                if time.time() > expires: raise ValueError('Declared witness-reader lifetime exceeded; no automatic retry')
                guard.check(full=False); time.sleep(10)
        for scope in SCOPES:
            path = RUN / (scope + '.json'); await_file(path); result = json.loads(path.read_text()); artifacts.append(path)
            if result['scope'] != scope or result['all_declared_scope_inputs_scanned'] is not True: raise ValueError('Complete original source scope required')
            legacy = None
            if scope not in SCOPES[:3]:
                folder = ROOT / 'data' / scope; legacy = json.loads((folder / 'corpus.json').read_text())['splits']['train']['shards']
                if result['counts']['input_tokens'] != source_plan['older_possible_source_tokens'][scope] or result['counts']['shards'] != len(legacy):
                    raise ValueError('Every declared original older source token/shard must be accounted for')
            elif scope == 'current_native':
                counts = result['counts']; expected = phase['complete_source_budget']
                if (counts['totals'] != dict(records=669863, input_tokens=577662698, response_or_next_token_targets=467581587)
                        or set(counts['families']) != set(expected)
                        or any(counts['families'][n] != dict(records=v['examples'], input_tokens=v['complete_context_tokens'],
                            response_or_next_token_targets=v['response_EOS_targets']) for n, v in expected.items())):
                    raise ValueError('Every complete original current family/token/target budget required')
            elif scope == 'future_human':
                expected = human.manifest['splits']['preferences.train']
                if result['counts'] != dict(original_pairs=85786, response_presentations=171572,
                        input_tokens=expected['packed_context_tokens'], response_EOS_targets=expected['response_EOS_targets']):
                    raise ValueError('Every complete original future human response and target required')
            elif result['counts'] != dict(records=3721, input_tokens=2287522, response_EOS_targets=584218):
                raise ValueError('Every complete original future code source/target required')
            counts = Counter(); local = []
            for identity, match in result['matches'].items():
                if identity not in questions: raise ValueError('Unexpected original pattern identity')
                question = questions[identity]; source = match['source']
                if source['scope'] != scope: raise ValueError('Exact original scope required in each source witness')
                key = (scope, json.dumps(source, sort_keys=True, separators=(',', ':')))
                if key not in context_cache:
                    if scope == 'current_native':
                        family, position = source['family'], source['position']; batch = data.batch(family, [position], 'cpu')
                        length = int(batch['attention_mask'][0].sum()); ids = batch['input_ids'][0, :length].tolist()
                        labels = batch['labels'][0, :length].tolist()
                        if length != data.lengths[family][position] or sum(v != -100 for v in labels[1:]) != data.targets[family][position]:
                            raise ValueError('Actual complete native training batch must match its source budget')
                    elif scope == 'future_human': ids, _ = human.sequence(source['pair'], source['field'])
                    elif scope == 'future_code':
                        if code.row(source['position'])['id'] != source['id']: raise ValueError('Exact original code source identity required')
                        ids, _ = code.sequence(source['position'])
                    else: ids, _ = reconstruct(ROOT / 'data' / scope, legacy, source)
                    digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()
                    text = tokenizer.decode(ids, skip_special_tokens=False)
                    for marker in ['<bos>', '<eos>', '<pad>']: text = text.replace(marker, '\0')
                    context_cache[key] = (normalized(text), digest, len(ids))
                text, digest, length = context_cache[key]; needle = normalized(question['text']); start = text.find(needle)
                if start < 0 or start != match['normalized_character_start']: raise ValueError('Recorded exact normalized signature does not occur at the actual whole source-record offset')
                counts['AI2D:' + question['kind']] += 1
                row = dict(id=question['case']['id'], original_question_id=question['case']['original_question_id'],
                    pattern_kind=question['kind'], scope=scope, original_source=source, independently_reconstructed_input_ids_sha256=digest,
                    whole_original_source_tokens=length, normalized_character_start=start, exact_matched_pattern=needle,
                    normalized_source_context_excerpt=text[max(0, start - 100):start + len(needle) + 100],
                    producer_projection_and_substring_automaton_not_used=True)
                local.append(row); witness_rows.append(row)
            if len(local) != result['matched_original_question_ids'] or dict(counts) != result['question_counts_by_benchmark']:
                raise ValueError('Every reported matched original question signature must rederive')
            scopes[scope] = dict(complete_original_token_accounting_verified=True, actual_matched_witnesses_reconstructed=len(local),
                matches_by_signature=dict(counts), input_token_count=(result['counts']['totals']['input_tokens'] if scope == 'current_native' else result['counts']['input_tokens']))
            output = OUT / (scope + '-actual-witnesses.jsonl')
            output.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in local)); artifacts.append(output)
            atomic_json(OUT / (scope + '-complete.json'), scopes[scope]); artifacts.append(OUT / (scope + '-complete.json'))
            print(json.dumps(dict(scope=scope, independent_witnesses=len(local), counts=dict(counts))), flush=True)
        complete_path = RUN / 'complete.json'; await_file(complete_path); complete = json.loads(complete_path.read_text()); artifacts.append(complete_path)
        qpath = RUN / 'original-question-pattern-results.jsonl'
        if (complete['all_six_declared_scopes_completed'] is not True or complete['plan_sha256'] != file_hash(source_plan_path)
                or complete['original_question_pattern_results_sha256'] != file_hash(qpath)):
            raise ValueError('Complete exact all-scope audit and every original case verdict required')
        records = list(map(json.loads, qpath.open())); artifacts.append(qpath)
        if [r['id'] for r in records] != [r['id'] for r in population.cases]: raise ValueError('Every original ordered question must be reconciled')
        matched = {(r['id'], r['pattern_kind'], r['scope']) for r in witness_rows}
        independently_rederived = {}
        for kind in KINDS:
            independently_rederived[kind] = dict(any_inspected_scope=sum(any((r['id'], kind, scope) in matched for scope in SCOPES) for r in records),
                by_scope={scope: sum((r['id'], kind, scope) in matched for r in records) for scope in SCOPES})
        for row in records:
            for kind in KINDS:
                expected = [scope for scope in SCOPES if (row['id'], kind, scope) in matched]
                if row['literal_text_patterns'][kind]['inspected_exact_match_scopes'] != expected or row['literal_text_patterns'][kind]['found'] != bool(expected):
                    raise ValueError('Every recorded original question/signature exposure flag must rederive')
        if complete['matched_questions_by_pattern'] != independently_rederived: raise ValueError('All original per-signature audit aggregates must independently reconstruct')
        guard.check(full=True)
        if torch.cuda.is_initialized(): raise ValueError('CPU witness reader initialized CUDA')
        atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, complete=True,
            original_AI2D_questions=3088, actual_matched_source_witnesses_reconstructed=len(witness_rows),
            total_declared_input_token_positions_accounted_for=sum(v['input_token_count'] for v in scopes.values()),
            independent_per_scope_accounting=scopes, independent_matched_questions_by_signature=independently_rederived,
            actual_current_batches_and_whole_older_records_used=True, producer_projection_and_automaton_not_used=True,
            negative_pools_not_independently_rescanned=True, semantic_image_and_complete_historical_independence_not_certified=True,
            source_files_sha256={str(p): file_hash(p) for p in artifacts},
            model_generations=0, model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), witnesses=len(witness_rows), model_updates=0)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__': main()
