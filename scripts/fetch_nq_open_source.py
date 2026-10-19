"""Retain original publisher NQ-Open question/answer files at an immutable revision."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.public_sources import fetch

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'google-research-datasets/natural-questions'


def key(question):
    return ' '.join(unicodedata.normalize('NFKC', question).casefold().split())


def main():
    output = ROOT / 'data/nq-open-original-source-v1'
    if output.exists(): raise ValueError('Preserve existing original NQ source; inspect before resuming')
    commit_raw = fetch('https://api.github.com/repos/' + REPOSITORY + '/commits/master', limit=2_000_000, timeout=20)
    commit = json.loads(commit_raw); revision = commit['sha']
    if len(revision) != 40 or any(value not in '0123456789abcdef' for value in revision): raise ValueError('Invalid publisher immutable revision')
    tree_raw = fetch('https://api.github.com/repos/' + REPOSITORY + '/git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1', limit=2_000_000, timeout=20)
    tree = json.loads(tree_raw)
    if tree['truncated']: raise ValueError('Publisher tree inventory incomplete')
    blobs = {entry['path']: entry for entry in tree['tree'] if entry['type'] == 'blob'}
    names = ['nq_open/README.md', 'LICENSE', 'nq_open/NQ-open.train.jsonl', 'nq_open/NQ-open.dev.jsonl']
    output.mkdir(); (output / 'publisher-commit.json').write_bytes(commit_raw); (output / 'publisher-tree.json').write_bytes(tree_raw)
    sources = {}
    for name in names:
        item = blobs[name]
        if item.get('size', 0) > 40_000_000: raise ValueError('Original source exceeds bounded input budget')
        url = 'https://raw.githubusercontent.com/' + REPOSITORY + '/' + revision + '/' + name
        raw = fetch(url, limit=40_000_000, timeout=20)
        actual_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if actual_blob != item['sha']: raise ValueError('Downloaded original source differs from publisher Git blob')
        path = output / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        sources[name] = {'url': url, 'bytes': len(raw), 'sha256': file_hash(path), 'publisher_git_blob_sha1': actual_blob}
        print('Original NQ publisher file retained', name, len(raw), flush=True)
    readme = (output / 'nq_open/README.md').read_text()
    if 'All of the Natural Questions data is released under the' not in readme or '[CC BY-SA 3.0]' not in readme:
        raise ValueError('Original publisher data-license declaration changed')
    stats = {}; groups = {}
    for split, count in [('train', 87925), ('dev', 3610)]:
        records = [json.loads(line) for line in (output / ('nq_open/NQ-open.' + split + '.jsonl')).read_text().splitlines()]
        if len(records) != count: raise ValueError('Original publisher NQ-open split count differs')
        questions = Counter()
        for row in records:
            if (set(row) != {'question', 'answer'} or not isinstance(row['question'], str) or not row['question'].strip()
                    or not isinstance(row['answer'], list) or not row['answer']
                    or any(not isinstance(answer, str) or not answer.strip() for answer in row['answer'])):
                raise ValueError('Original publisher question/answer schema differs')
            questions[key(row['question'])] += 1
        groups[split] = set(questions)
        stats[split] = {'original_rows': len(records), 'canonical_questions': len(questions),
                        'repeated_question_rows': sum(value - 1 for value in questions.values())}
    manifest = {'retrieved_at': datetime.now(timezone.utc).isoformat(), 'repository': REPOSITORY, 'revision': revision,
                'sources': sources, 'splits': stats, 'exact_canonical_train_dev_question_overlap': len(groups['train'] & groups['dev']),
                'annotation_origin': 'Publisher NQ-open conversion of original Natural Questions human annotations; original answer aliases unchanged.',
                'data_license': 'CC BY-SA 3.0', 'data_license_source': sources['nq_open/README.md']['url'],
                'repository_code_license': 'Apache-2.0; separate from the explicit data license',
                'implementation_sha256': file_hash(Path(__file__)), 'external_model_teacher_used': False,
                'training_started': False, 'goal_achieved': False,
                'scope': 'Original source retrieval/schema/license and publisher blob verification only. No train admission, independent contamination audit or model capability claim.',
                'limitations': ['Original NQ-open files carry questions and reference strings but omit original HTML/photo/article identity.',
                                'Question duplicates/official split overlaps must be handled before training; semantic paraphrases are not audited here.',
                                'NQ-open is an English open-domain QA source, not comprehensive professional or all-domain expert coverage.']}
    atomic_json(output / 'manifest.json', manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != 'sources'}), flush=True)


if __name__ == '__main__': main()
