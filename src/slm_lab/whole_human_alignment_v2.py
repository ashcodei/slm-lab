"""Whole public human demonstrations/rankings, with native packed responses.

Original held trees and all original HH test prompts are protected. Exact
question screening does not certify semantic independence or factual accuracy.
HH responses remain other-model-generated and human-ranked, not human-written.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata

import ahocorasick
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .code_tasks import TASKS
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .diverse_images import atomic_json
from .human_dialogue_corpus import examples, REVISION, TREE_FILE, TREE_SHA256, VALID_FILE, VALID_SHA256
from .model import ModelConfig

FORMAT = 'whole-original-human-conversation-and-ranking-packed-v2'


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


class WholeResponseEncoder:
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config
        if (tokenizer.get_vocab_size() != config.vocab_size or config.vocab_size > 65536
                or [tokenizer.token_to_id(k) for k in ('<pad>', '<bos>', '<eos>')] != [0, 1, 2]):
            raise ValueError('Exact native own vocabulary and BOS/EOS/PAD framing required')

    def sequence(self, row, field):
        prompt = row['prompt']
        if not prompt.strip() or not isinstance(row[field], str) or not row[field].strip():
            raise ValueError('Complete nonempty original prompt and response required')
        framed = prompt if prompt.rstrip().endswith('Assistant:') else 'User: '+prompt+'\nAssistant:'
        prefix = [1, *self.tokenizer.encode(framed).ids]
        response = [*self.tokenizer.encode(row[field]).ids, 2]
        ids = prefix + response
        if len(ids) > self.config.max_text_length:
            raise ValueError('Whole prompt plus response/EOS exceeds the actual full context')
        return ids, len(prefix)


class NamedHeldQuestions:
    def __init__(self, root):
        self.files, self.counts = {}, {}
        self.matcher = ahocorasick.Automaton()
        for family, relative, field in [('knowledge','data/knowledge-eval-v1/questions.jsonl','question'),
            ('math','data/math-reasoning-v2/test.jsonl','question'),
            ('public_code','data/public-code-assessment-v1/questions.jsonl','specification')]:
            path = root/relative; self.files[str(path)] = file_hash(path); self.counts[family] = 0
            for row in rows(path):
                text = normalized(row[field])
                if not text:
                    raise ValueError('Every full original protected question must be present')
                self.matcher.add_word(text, family); self.counts[family] += 1
        paths = sorted((root/'data/public-apps-publisher-test-evaluation-v1/original-test').glob('*/question.txt'))
        if len(paths) != 5000 or self.counts != dict(knowledge=559, math=1319, public_code=175):
            raise ValueError('Complete actual original named assessment populations required')
        for path in paths:
            self.files[str(path)] = file_hash(path)
            self.matcher.add_word(normalized(path.read_text()), 'original_APPS_test')
        self.counts['original_APPS_test'] = len(paths); self.counts['local_code'] = len(TASKS)
        for task in TASKS:
            self.matcher.add_word(normalized(task[2]), 'local_code')
        self.matcher.make_automaton()

    def matches(self, text):
        return sorted({value for _, value in self.matcher.iter(normalized(text))})


def split_human_ranked_pair(raw):
    marker = '\n\nAssistant:'
    chosen, rejected = raw['chosen'], raw['rejected']
    a, b = chosen.rfind(marker), rejected.rfind(marker)
    if a < 0 or b < 0 or chosen[:a+len(marker)] != rejected[:b+len(marker)]:
        raise ValueError('Original paired responses must share the complete final source prompt')
    prompt = chosen[:a+len(marker)]; good, bad = chosen[a+len(marker):], rejected[b+len(marker):]
    if not good.strip() or not bad.strip() or good.strip() == bad.strip():
        raise ValueError('Original human-ranked responses must be nonempty and distinct')
    # Preserve even the original response whitespace; no suffix is truncated.
    return dict(prompt=prompt, chosen=good, rejected=bad)


def hh_records(source, manifest, split):
    for blob in manifest['sources']:
        if blob['original_split'] != split:
            continue
        count = 0
        with gzip.open(source/blob['file'], 'rt') as stream:
            for position, line in enumerate(stream):
                count += 1
                yield blob, position, json.loads(line)
        if count != blob['complete_original_rows']:
            raise ValueError('Every original human-ranking source row must be accounted for')


class PackedWriter:
    def __init__(self, root, name):
        self.index = (root/(name+'.jsonl')).open('x')
        self.tokens = (root/(name+'.tokens.u16')).open('xb')
        self.count = self.positions = self.targets = self.max_context = self.max_response = 0
        self.languages = Counter()

    def append(self, row, sequences):
        native = {}
        for field, (ids, start) in sequences.items():
            if not 1 <= start < len(ids) <= 4096 or ids[0] != 1 or ids[-1] != 2:
                raise ValueError('Whole native prefix and response/EOS must fit the actual context')
            self.tokens.write(np.asarray(ids, dtype='<u2').tobytes())
            native[field] = dict(offset=self.positions, length=len(ids), target_start=start)
            self.positions += len(ids); self.targets += len(ids)-start
            self.max_context = max(self.max_context, len(ids)); self.max_response = max(self.max_response, len(ids)-start)
        self.index.write(json.dumps({**row, 'native': native}, ensure_ascii=False)+'\n')
        self.count += 1
        if row.get('language'):
            self.languages[row['language']] += 1

    def close(self):
        self.index.close(); self.tokens.close()
        return dict(examples=self.count, packed_context_tokens=self.positions, response_EOS_targets=self.targets,
                    maximum_whole_context=self.max_context, maximum_whole_response_EOS=self.max_response,
                    languages=dict(self.languages))


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve previous preparation and evidence; a fresh output is required')
    source = root/'data/openassistant-source-v1'; hh = root/'data/original-HH-alignment-v2'
    token = root/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
    config_path = root/'configs/model.json'; config = ModelConfig.load(config_path)
    encoder = WholeResponseEncoder(Tokenizer.from_file(str(token)), config)
    protection = NamedHeldQuestions(root)
    files = dict(protection.files)
    for name, expected in [(TREE_FILE, TREE_SHA256), (VALID_FILE, VALID_SHA256)]:
        if file_hash(source/name) != expected:
            raise ValueError('Original OpenAssistant source bytes must match the pinned publisher receipt')
        files[str(source/name)] = expected
    readme = source/'README.md'
    if file_hash(readme) != '52bd472cf3cae04a4b21ecd5316f929741c47d991d4763b041c3a94de072dc31':
        raise ValueError('Exact retained OpenAssistant license/attribution declaration required')
    files[str(readme)] = file_hash(readme)
    human_ranked = json.loads((hh/'manifest.json').read_text())
    if human_ranked['source_revision'] != 'c72f5cee8eb7b4d2ea5617657f4430d5e333af07' or human_ranked['license'] != 'MIT':
        raise ValueError('Exact public human-ranking revision and license required')
    collection_plan = json.loads((hh/'plan.json').read_text())
    old_manifest = root/'data/alignment-public-v1/manifest.json'
    if (human_ranked['plan_sha256'] != file_hash(hh/'plan.json')
            or collection_plan['original_manifest_sha256'] != file_hash(old_manifest)):
        raise ValueError('Original retained human-ranking collection receipts required')
    for name, sha in human_ranked['files_sha256'].items():
        if file_hash(hh/name) != sha:
            raise ValueError('Complete original human-ranking blobs changed')
        files[str(hh/name)] = sha
    for path in (hh/'manifest.json', hh/'plan.json', old_manifest, token, config_path, Path(__file__).resolve(),
                 root/'src/slm_lab/human_dialogue_corpus.py', root/'src/slm_lab/code_tasks.py', root/'data/human-dialogue-v1/train.jsonl'):
        files[str(path)] = file_hash(path)
    held_pairs = set(); test_source_counts = Counter()
    for blob, position, raw in hh_records(hh, human_ranked, 'test'):
        test_source_counts[blob['subset']] += 1
        try:
            held_pairs.add(normalized(split_human_ranked_pair(raw)['prompt']))
        except ValueError:
            # Even malformed comparison tails protect any common original
            # final assistant prompt if one exists; reviews below retain them.
            for field in ('chosen', 'rejected'):
                mark = raw[field].rfind('\n\nAssistant:')
                if mark >= 0:
                    held_pairs.add(normalized(raw[field][:mark+len('\n\nAssistant:')]))
    prior_prompts = set(held_pairs)
    for relative in ('data/alignment-curriculum-v1/instruction.valid.jsonl',
                     'data/human-dialogue-v1/valid.jsonl', 'data/human-preference-curriculum-v1/preferences.valid.jsonl'):
        path = root/relative; files[str(path)] = file_hash(path)
        prior_prompts.update(normalized(row['prompt']) for row in rows(path))
    valid_ids = set(pq.read_table(source/VALID_FILE, columns=['message_tree_id']).column('message_tree_id').to_pylist())
    with gzip.open(source/TREE_FILE, 'rt') as stream:
        trees = [json.loads(line) for line in stream]
    ids = {tree['message_tree_id'] for tree in trees}
    if len(ids) != len(trees) or not valid_ids <= ids or None in valid_ids:
        raise ValueError('Unique original trees and complete official validation roots required')
    valid_roots = {normalized(tree['prompt']['text']) for tree in trees if tree['message_tree_id'] in valid_ids}
    trees.sort(key=lambda tree:(tree['message_tree_id'] not in valid_ids, tree['message_tree_id']))
    guard = FileBindings(files)
    output.mkdir()
    atomic_json(output/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(), format=FORMAT,
        source_files_sha256=files, config=asdict(config), protected_named_questions=protection.counts,
        original_validation_trees=len(valid_ids), all_original_HH_test_prompt_groups=len(held_pairs),
        original_HH_test_rows=dict(test_source_counts), original_HH_complete_sources=True,
        no_truncation=True, maximum_whole_context=config.max_text_length, no_legacy_half_window_cap=True,
        public_human_ranked_response_origin=human_ranked['response_origin'], source_revision=REVISION,
        training_admitted=False, goal_achieved=False))
    writers = {name:PackedWriter(output, name) for name in ('dialogue.valid','dialogue.train','preferences.valid','preferences.train')}
    exclusions = Counter(); reviews = (output/'reviews.jsonl').open('x')
    def review(family, role, identifier, reason, **details):
        exclusions[family+':'+reason] += 1
        reviews.write(json.dumps(dict(family=family, role=role, id=identifier, reason=reason, **details))+'\n')
    seen_dialogue=set(); valid_dialogue_prompts=set(); kept_groups={'train':set(),'valid':set()}
    prior_ids = {row['id'] for row in rows(root/'data/human-dialogue-v1/train.jsonl')}
    prior_valid_ids = {row['id'] for row in rows(root/'data/human-dialogue-v1/valid.jsonl')}
    recovery={'train':Counter(),'valid':Counter()}; protected_prior=prior_prompts.copy()
    try:
        for tree in trees:
            identifier=tree['message_tree_id'];role='valid' if identifier in valid_ids else 'train'
            if role=='train' and (normalized(tree['prompt']['text']) in valid_roots or normalized(tree['prompt']['text']) in prior_prompts):
                review('dialogue',role,identifier,'protected original/prior validation root');continue
            selected, issues=examples(tree)
            for issue in issues:
                review('dialogue',role,issue.get('message_id',identifier),issue['reason'],tree_id=identifier)
            hits=sorted({hit for row in selected for hit in protection.matches(row['prompt']+' '+row['response'])})
            if hits:
                review('dialogue',role,identifier,'named held question text',families=hits);continue
            for row in selected:
                key=(row['prompt'],row['response']);prompt=normalized(row['prompt'])
                if role=='train' and (prompt in prior_prompts or prompt in valid_dialogue_prompts):
                    review('dialogue',role,row['id'],'protected complete held prompt');continue
                if key in seen_dialogue:
                    review('dialogue',role,row['id'],'exact full prompt/response duplicate');continue
                try:
                    sequence=encoder.sequence(row,'response')
                except ValueError as error:
                    review('dialogue',role,row['id'],str(error));continue
                seen_dialogue.add(key)
                if role=='valid':
                    valid_dialogue_prompts.add(prompt)
                kept_groups[role].add(row['group_id'])
                target_count=len(sequence[0])-sequence[1]
                recovery[role]['retained_prior_examples' if row['id'] in (prior_ids if role=='train' else prior_valid_ids) else 'new_complete_source_examples']+=1
                recovery[role]['responses_over_legacy_half_window']+=int(target_count>config.max_text_length//2)
                writers['dialogue.'+role].append(dict(row,split=role,target_tokens=target_count),{'response':sequence})
        protected_prior.update(valid_dialogue_prompts)
        seen_comparisons=set(); kept_prompt_groups={'train':set(),'valid':set()};source_rows_count=Counter()
        for original_split,role in [('test','valid'),('train','train')]:
            for blob,position,raw in hh_records(hh,human_ranked,original_split):
                source_rows_count[original_split]+=1
                identifier=f"HH:{blob['subset']}:{original_split}:{position}"
                try:
                    pair=split_human_ranked_pair(raw)
                except ValueError as error:
                    review('preferences',role,identifier,str(error));continue
                prompt=normalized(pair['prompt'])
                if role=='train' and (prompt in protected_prior or prompt in kept_prompt_groups['valid']):
                    review('preferences',role,identifier,'protected original/prior held prompt');continue
                hits=protection.matches(pair['prompt']+' '+pair['chosen']+' '+pair['rejected'])
                if hits:
                    review('preferences',role,identifier,'named held question text',families=hits);continue
                key=hashlib.sha256(json.dumps(pair,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                if key in seen_comparisons:
                    review('preferences',role,identifier,'exact whole comparison duplicate');continue
                try:
                    sequences={field:encoder.sequence(pair,field) for field in ('chosen','rejected')}
                except ValueError as error:
                    review('preferences',role,identifier,str(error));continue
                seen_comparisons.add(key);kept_prompt_groups[role].add(prompt)
                row=dict(pair,id=identifier,source_prompt_id=hashlib.sha256(' '.join(pair['prompt'].casefold().split()).encode()).hexdigest(),
                    split=role,original_split=original_split,source_blob=blob['file'],original_source_position=position,
                    source='GitHub/anthropics/hh-rlhf/'+blob['subset'],source_url=blob['url'],revision=human_ranked['source_revision'],
                    feedback_source='human',response_origin=human_ranked['response_origin'],license='MIT',
                    source_record_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                    whole_original_response_whitespace_preserved=True,
                    PPO_prompt_reserves_256_tokens=sequences['chosen'][1]+256<=config.max_text_length)
                writers['preferences.'+role].append(row,sequences)
                if source_rows_count[original_split]%2000==0:
                    atomic_json(output/'progress.json',dict(status='packing whole public human-ranked source',original_split=original_split,
                        original_source_rows=source_rows_count[original_split],retained=writers['preferences.'+role].count,goal_achieved=False))
        if kept_groups['train'] & kept_groups['valid'] or kept_prompt_groups['train'] & kept_prompt_groups['valid']:
            raise ValueError('Original held tree/prompt groups must never enter training')
    finally:
        stats={name:writer.close() for name,writer in writers.items()};reviews.close()
    guard.check(full=True)
    manifest=dict(completed_at=datetime.now(timezone.utc).isoformat(),format=FORMAT,plan_sha256=file_hash(output/'plan.json'),
        tokenizer_sha256=file_hash(token),config=asdict(config),splits=stats,original_source_trees=len(trees),
        original_validation_trees=len(valid_ids),original_HH_rows=dict(source_rows_count),
        all_original_HH_test_prompt_groups_protected=len(held_pairs),dialogue_source_recovery={k:dict(v) for k,v in recovery.items()},
        excluded=dict(exclusions),original_tree_overlap=0,original_preference_prompt_overlap=0,
        source_files_sha256=files,files_sha256={p.name:file_hash(p) for p in output.iterdir() if p.suffix in ('.jsonl','.u16')},
        full_original_responses_and_EOS_preserved=True,no_half_window_response_cap=True,source_human_review_filter_unchanged=True,
        synthetic_human_demonstration_responses_admitted=0,human_ranked_response_origin=human_ranked['response_origin'],
        external_teacher_weights=False,training_admitted=False,goal_achieved=False,
        limitations=['Publisher human/review flags and original human rankings are source evidence, not independently verified authorship or factual/algorithmic correctness.',
            'Exact original held question/prompt/group screening does not certify semantic independence or remove earlier model exposure.',
            'HH answers are other-model-generated and human-ranked, not own responses with fresh human labels.',
            'Selection/preparation is not training, preference generalization, alignment quality or high-end capability.'])
    atomic_json(output/'manifest.json',manifest)
    return manifest
