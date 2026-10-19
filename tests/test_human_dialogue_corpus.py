from copy import deepcopy
import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers

from slm_lab.corpus import file_hash
import slm_lab.human_dialogue_corpus as corpus

from slm_lab.human_dialogue_corpus import examples


def node(identifier, role, text, replies=(), rank=None, **extra):
    return {'message_id': identifier, 'role': role, 'text': text, 'lang': 'en',
            'synthetic': False, 'model_name': None, 'deleted': False, 'review_result': True,
            'rank': rank, 'labels': {'quality': {'value': .9, 'count': 3}}, 'replies': list(replies), **extra}


def test_human_paths_keep_exact_turns_and_prune_synthetic_ancestors():
    follow = node('follow', 'prompter', 'Now explain the units.', [node('second', 'assistant', 'Meters per second.', rank=0)])
    good = node('first', 'assistant', 'Speed is distance over time.', [follow], rank=0)
    bad = node('model', 'assistant', 'Model-generated response.', [deepcopy(follow)], rank=0,
               synthetic=True, model_name='external-model')
    tree = {'message_tree_id': 'root', 'tree_state': 'ready_for_export',
            'prompt': node('root', 'prompter', 'Explain speed.', [good, bad])}
    rows, rejected = examples(tree)
    assert [row['source_message_id'] for row in rows] == ['first', 'second']
    assert rows[1]['prompt'] == 'User: Explain speed.\nAssistant: Speed is distance over time.\nUser: Now explain the units.\nAssistant:'
    assert rows[1]['response'] == 'Meters per second.' and rows[1]['group_id'] == 'oasst2:root'
    assert all(row['synthetic'] is False for row in rows)
    assert any(issue['reason'] == 'synthetic/model/unknown authorship' for issue in rejected)


def test_only_top_ranked_or_single_reviewed_replies_are_selected():
    replies = [node('best', 'assistant', 'Best human response.', rank=0),
               node('second', 'assistant', 'Lower-ranked response.', rank=1),
               node('unknown', 'assistant', 'Unranked response.')]
    tree = {'message_tree_id': 'root', 'tree_state': 'ready_for_export',
            'prompt': node('root', 'prompter', 'Question.', replies)}
    rows, rejected = examples(tree)
    assert [row['source_message_id'] for row in rows] == ['best'] and len(rejected) == 2
    tree['prompt']['replies'] = [node('single', 'assistant', 'Only reviewed response.')]
    assert examples(tree)[0][0]['source_message_id'] == 'single'


def test_failed_review_or_positive_exclusion_label_prunes_the_entire_path():
    child = node('answer', 'assistant', 'A response.', rank=0)
    root = node('root', 'prompter', 'A prompt.', [child])
    root['labels']['pii'] = {'value': .3, 'count': 3}
    rows, _ = examples({'message_tree_id': 'root', 'tree_state': 'ready_for_export', 'prompt': root})
    assert rows == []
    root['labels'].pop('pii'); root['review_result'] = False
    assert examples({'message_tree_id': 'root', 'tree_state': 'ready_for_export', 'prompt': root})[0] == []


def test_preparation_preserves_official_trees_and_excludes_reserved_or_truncated_examples(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()

    def tree(identifier, question, response):
        return {'message_tree_id': identifier, 'tree_state': 'ready_for_export',
                'prompt': node(identifier, 'prompter', question,
                               [node(identifier + '-reply', 'assistant', response, rank=0)])}

    records = [tree('valid', 'Protected official prompt.', 'Original validation response.'),
               tree('copy', 'Protected official prompt.', 'A different response.'),
               tree('train', 'New request.', 'Original human response.'),
               tree('reserved', 'This exact reserved question should not appear.', 'A response.'),
               tree('long', 'A long request.', 'word ' * 4000)]
    with gzip.open(source / corpus.TREE_FILE, 'wt') as stream:
        stream.write(''.join(json.dumps(record) + '\n' for record in records))
    pq.write_table(pa.table({'message_tree_id': ['valid']}), source / corpus.VALID_FILE)
    (source / 'README.md').write_text('license: apache-2.0\n')
    monkeypatch.setattr(corpus, 'TREE_SHA256', file_hash(source / corpus.TREE_FILE))
    monkeypatch.setattr(corpus, 'VALID_SHA256', file_hash(source / corpus.VALID_FILE))
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    vocab.update({f'unused-{i}': i for i in range(4, 32768)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / 'tokenizer.json'; tokenizer.save(str(tokenizer_path))
    question_path = tmp_path / 'questions.jsonl'
    question_path.write_text(json.dumps({'id': 'reserved', 'question': records[3]['prompt']['text']}) + '\n')
    empty = tmp_path / 'empty.jsonl'; empty.write_text('')
    output = tmp_path / 'prepared'
    result = corpus.prepare(source, output, tokenizer_path, question_path, [empty], empty, empty)
    assert result['splits']['train']['examples'] == result['splits']['valid']['examples'] == 1
    assert result['tree_overlap'] == result['prompt_overlap'] == 0
    assert result['excluded']['protected official/prior validation prompt'] == 1
    assert result['excluded']['reserved knowledge/math/public-code assessment text'] == 1
    assert result['excluded']['would truncate conversation/response'] == 1
    train = json.loads((output / 'train.jsonl').read_text()); valid = json.loads((output / 'valid.jsonl').read_text())
    assert train['source_message_tree_id'] == 'train' and valid['source_message_tree_id'] == 'valid'
    assert train['response'] == 'Original human response.' and valid['response'] == 'Original validation response.'
    assert corpus.prepare(source, output, tokenizer_path, question_path, [empty], empty, empty) == result
